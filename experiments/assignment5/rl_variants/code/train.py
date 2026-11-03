"""Frozen three-shot A5 experiments, separate trainer and NCCL-synchronized sampler."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import signal
import time

from common import write_json, append_json, sha, data_rows, stats_for_records
from objectives import advantages_for_method, loss_for_method


def main():
    import numpy as np
    import torch
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from reference.grpo import pack_prompt_and_response_ids, get_response_log_probs
    from reference.drgrpo_grader import r1_zero_reward_fn
    from reference.vllm_utils import _http_json
    from server import Sampler
    from scoring import response_log_probs

    code = Path(__file__).resolve().parent
    root = code.parent
    protocol = json.loads((root / "protocol.json").read_text())
    methods = {s["name"]: s for s in protocol["methods_in_order"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=methods, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "pilot", "formal"], default="formal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--val-examples", type=int)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--old-log-probs-batch-size", type=int, default=8)
    parser.add_argument("--scorer", choices=["reference", "response-only"], default="reference")
    parser.add_argument("--head-chunk-size", type=int, default=512)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()
    settings = methods[args.method]
    off = args.method.startswith("offpolicy_")
    steps = args.steps or {"smoke": 2, "pilot": 50, "formal": 200}[args.profile]
    val_count = args.val_examples or (32 if args.profile == "smoke" else 1024)
    if off and args.microbatch_size != 8:
        parser.error("Off-policy uses eight responses per independent optimizer update")
    if min(args.microbatch_size, args.old_log_probs_batch_size, args.head_chunk_size) < 1:
        parser.error("Batch sizes and projection chunk size must be positive")
    c = protocol["common"]
    output = args.output.resolve()
    if (output / "config.json").exists():
        raise FileExistsError(f"Existing run is preserved: {output}")
    output.mkdir(parents=True, exist_ok=True)
    spec = json.loads((code / "env-spec.json").read_text())
    model_path = spec["model"]
    train_path, val_path = root / "data/gsm8k/train.jsonl", root / "data/gsm8k/test.jsonl"
    train_data, val_data = data_rows(train_path, 6400), data_rows(val_path, val_count)
    template = (code / "prompt.prompt").read_text()
    assert template.count("User:") == 4 and template.count("{question}") == 1
    assert not any(r["question"] in template for r in train_data + val_data)
    config = {**c, **settings, "profile": args.profile, "seed": args.seed,
              "rollout_steps": steps, "val_examples": val_count, "microbatch_size": args.microbatch_size,
              "train_ids": [r["id"] for r in train_data], "val_ids": [r["id"] for r in val_data],
              "prompt_sha256": sha(code / "prompt.prompt"), "train_sha256": sha(train_path),
              "val_sha256": sha(val_path), "protocol_sha256": sha(root / "protocol.json"),
              "model_manifest_sha256": sha(code / "model-manifest.json"),
              "code_sha256": {p.name: sha(p) for p in code.glob("*.py")},
              "entropy_population": "all responses" if off or args.method == "grpo" else "nonzero-advantage responses",
              "zero_advantage_pruning": not off and args.method != "grpo",
              "precision": "FP32 parameters and AdamW states; BF16 autocast and inference",
              "checkpoint_saved": args.save_model,
              "scorer": args.scorer, "head_chunk_size": args.head_chunk_size,
              "gradient_checkpointing": not args.no_gradient_checkpointing,
              "old_log_probs_batch_size": args.old_log_probs_batch_size,
              "gradient_accumulation_steps": 1 if off else (255 // args.microbatch_size + 1),
              "gradient_accumulation_note": "nominal full rollout; pruning may reduce backward calls while retaining original denominator",
              "performance_policy_sha256": sha(root / "performance-policy.json") if (root / "performance-policy.json").exists() else None,
              "loss_normalization_constant": 256 * 512 if settings["loss_normalization"] == "constant" else None}
    write_json(output / "config.json", config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    updates = 0
    run = None
    sampler = None
    completed = False
    started = time.monotonic()
    best = 0.0
    signal_updates = 0
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    def saved_records(directory, step, records):
        path = output / directory / f"step-{step:04d}.json.gz"
        path.parent.mkdir(exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=5) as stream:
            json.dump(records, stream, ensure_ascii=False)
    def log(payload):
        append_json(output / "metrics.jsonl", payload)
        run.log(payload)
    try:
        run = wandb.init(entity=c["wandb_entity"], project=c["wandb_project"],
                         group="a5-3shot-20261029", job_type=args.profile,
                         name=f"{args.method}-{args.profile}-seed{args.seed}",
                         config=config, dir=str(output), mode="online")
        run.define_metric("rollout_step")
        for namespace in ["train", "val", "system", "debug"]:
            run.define_metric(f"{namespace}/*", step_metric="rollout_step")
        write_json(output / "wandb-run.json", {"id": run.id, "url": run.url})
        print("TRAIN_RUN", run.url, flush=True)
        policy = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32,
                                                     attn_implementation="sdpa").to(device)
        policy.config.use_cache = False
        if not args.no_gradient_checkpointing:
            policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        def score(ids, labels, mask, entropy=False):
            if args.scorer == "response-only":
                return response_log_probs(policy, ids, labels, mask, entropy, args.head_chunk_size)
            return get_response_log_probs(policy, ids, labels, entropy)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5, betas=(.9, .95),
                                      weight_decay=0., foreach=False)
        probe_parameter = next(p for name, p in reversed(list(policy.named_parameters())) if "norm" in name)
        def probe():
            return hashlib.sha256(probe_parameter.detach().float().cpu().numpy().tobytes()).hexdigest()
        sampler = Sampler(model_path, output, seed=args.seed)
        sampler.start()
        sampler.init_weight_sync("cuda:0")
        def generate(examples, n, seed):
            ids = [tokenizer.encode(template.format(question=r["question"]),
                                    add_special_tokens=False) for r in examples]
            assert max(map(len, ids)) + 512 <= 2048
            records = []
            for start in range(0, len(examples), 128):
                response = _http_json("POST", sampler.base_url + "/v1/completions",
                    {"model": model_path, "prompt": ids[start:start + 128], "n": n,
                     "temperature": 1., "top_p": 1., "top_k": -1, "min_p": 0.,
                     "max_tokens": 512, "seed": seed, "stop": ["</answer>"],
                     "include_stop_str_in_output": True, "return_token_ids": True}, timeout=900)
                choices = sorted(response["choices"], key=lambda x: x["index"])
                assert len(choices) == len(examples[start:start + 128]) * n
                for j, example in enumerate(examples[start:start + 128]):
                    for k in range(n):
                        item = choices[j * n + k]
                        tokens = item.get("token_ids")
                        assert tokens and len(tokens) <= 512
                        records.append({**example, "sample_index": k,
                            "prompt_token_ids": ids[start + j], "response": item["text"],
                            "response_token_ids": tokens, "finish_reason": item.get("finish_reason"),
                            "rewards": r1_zero_reward_fn(item["text"], example["ground_truth"])})
            return records
        def evaluate(step):
            nonlocal best
            t = time.monotonic()
            records = generate(val_data, 1, 100000)
            saved_records("validation", step, records)
            metrics = stats_for_records(records)
            best = max(best, metrics["reward"])
            log({"rollout_step": step, "optimizer_updates": updates,
                 **{f"val/{k}": v for k, v in metrics.items()},
                 "system/eval_seconds": time.monotonic() - t,
                 "system/cumulative_wall_seconds": time.monotonic() - started})
            run.log({"rollout_step": step, "samples/validation": wandb.Table(
                columns=["id", "response", "reward", "tokens"],
                data=[[r["id"], r["response"], r["rewards"]["reward"],
                       len(r["response_token_ids"])] for r in records[:12]])})
            print("EVAL", step, metrics, flush=True)
            return metrics
        sampler.sync_policy_weights(policy)
        initial = evaluate(0)
        initial_probe = probe()
        for step in range(1, steps + 1):
            iteration_started = time.monotonic()
            phase_started = time.monotonic()
            if step > 1:
                sampler.sync_policy_weights(policy)
            sync_seconds = time.monotonic() - phase_started
            phase_started = time.monotonic()
            records = generate(rng.sample(train_data, 32), 8, args.seed + step * 100)
            generation_seconds = time.monotonic() - phase_started
            saved_records("rollouts", step, records)
            rewards = torch.tensor([r["rewards"]["reward"] for r in records])
            advantages = advantages_for_method(rewards, 8, settings)
            pack = pack_prompt_and_response_ids([r["prompt_token_ids"] for r in records],
                       [r["response_token_ids"] for r in records], tokenizer.pad_token_id)
            for i, r in enumerate(records):
                assert pack["labels"][i][pack["response_mask"][i]].tolist() == r["response_token_ids"]
            old = None
            old_hash = None
            phase_started = time.monotonic()
            if off:
                policy.eval()
                old = torch.empty(pack["labels"].shape)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for start in range(0, 256, args.old_log_probs_batch_size):
                        stop = start + args.old_log_probs_batch_size
                        result = score(pack["input_ids"][start:stop].to(device),
                                       pack["labels"][start:stop].to(device),
                                       pack["response_mask"][start:stop].to(device))
                        old[start:stop] = result["log_probs"].float().cpu()
                        del result
                old_hash = hashlib.sha256(old.numpy().tobytes()).hexdigest()
                path = output / "old_log_probs"
                path.mkdir(exist_ok=True)
                torch.save({"log_probs": old, "advantages": advantages,
                            "response_mask": pack["response_mask"], "sha256": old_hash}, path / f"step-{step:04d}.pt")
            old_logprob_seconds = time.monotonic() - phase_started
            indices = list(range(256))
            random.Random(args.seed + 1000000 + step).shuffle(indices)
            if config["zero_advantage_pruning"]:
                indices = [i for i in indices if advantages[i] != 0]
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            before = probe()
            update_start = updates
            total_loss = 0.
            entropy_sum = 0.
            entropy_count = 0
            norms = []
            stats = []
            torch.cuda.reset_peak_memory_stats()
            train_started = time.monotonic()
            for start in range(0, len(indices), args.microbatch_size):
                selected = torch.tensor(indices[start:start + args.microbatch_size])
                mask = pack["response_mask"][selected].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = score(pack["input_ids"][selected].to(device),
                                   pack["labels"][selected].to(device), mask, True)
                    loss, meta = loss_for_method(result["log_probs"],
                        old[selected].to(device) if off else None, advantages[selected].to(device),
                        mask, settings, full_count=256)
                    if not off and settings["loss_normalization"] == "sequence":
                        loss = loss * (len(selected) / 256)
                    entropy_sum += float(result["token_entropy"].detach().masked_fill(~mask, 0.).sum())
                    entropy_count += int(mask.sum())
                loss.backward()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                total_loss += float(loss.detach())
                stats.append((len(selected), meta))
                if off:
                    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    norms.append(float(norm))
                    append_json(output / "updates.jsonl", {"rollout_step": step,
                        "optimizer_update": updates, "sample_indices": selected.tolist(),
                        "loss": float(loss.detach()), "grad_norm": float(norm), **meta})
                del result, loss
            if not off:
                if not indices:
                    for p in policy.parameters():
                        if p.requires_grad:
                            p.grad = torch.zeros_like(p)
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                norms.append(float(norm))
                append_json(output / "updates.jsonl", {"rollout_step": step,
                    "optimizer_update": updates, "loss": total_loss,
                    "grad_norm": float(norm), "active_sequences": len(indices)})
            expected = 32 if off else 1
            assert updates - update_start == expected
            if off:
                assert hashlib.sha256(old.numpy().tobytes()).hexdigest() == old_hash
            changed = before != probe()
            signal_updates += int(max(norms) > 0 and changed)
            unit_weights = [(n if args.method == "offpolicy_gspo" else m["response_tokens"], m)
                            for n, m in stats]
            clip_fraction = sum(w * m["clip_fraction"] for w, m in unit_weights) / max(sum(w for w, _ in unit_weights), 1)
            payload = {"rollout_step": step, "optimizer_updates": updates,
                **{f"train/{k}": v for k, v in stats_for_records(records, 8).items()},
                "train/loss": total_loss / 32 if off else total_loss,
                "train/grad_norm": float(np.mean(norms)), "train/max_grad_norm": max(norms),
                "train/token_entropy": entropy_sum / entropy_count if entropy_count else None,
                "train/entropy_tokens": entropy_count, "train/active_sequence_fraction": float((advantages != 0).float().mean()),
                "train/clip_fraction": clip_fraction,
                "train/max_ratio": max((m["ratio_max"] for _, m in stats), default=1.),
                "system/step_seconds": time.monotonic() - iteration_started,
                "system/train_seconds": time.monotonic() - train_started,
                "system/generation_seconds": generation_seconds,
                "system/weight_sync_seconds": sync_seconds,
                "system/old_logprob_seconds": old_logprob_seconds,
                "system/cumulative_wall_seconds": time.monotonic() - started,
                "system/actor_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                "debug/probe_changed": int(changed), "debug/updates_this_rollout": expected,
                "debug/backward_microbatches": (len(indices) + args.microbatch_size - 1) // args.microbatch_size}
            log(payload)
            write_json(output / "progress.json", {"status": "training", "rollout_step": step,
                "target_steps": steps, "optimizer_updates": updates, "wandb_url": run.url, "metrics": payload})
            print("TRAIN", args.method, step, "/", steps, "reward", payload["train/reward"],
                  "grad", payload["train/grad_norm"], "seconds", payload["system/step_seconds"], flush=True)
            if step == 1 or step % 40 == 0:
                run.log({"rollout_step": step, "samples/train": wandb.Table(
                    columns=["id", "response", "reward"],
                    data=[[r["id"], r["response"], r["rewards"]["reward"]] for r in records[:16]])})
            del old, pack, advantages
            if step % 10 == 0 or step == steps:
                sampler.sync_policy_weights(policy)
                final = evaluate(step)
        assert signal_updates > 0, "No real nonzero-gradient parameter update was observed"
        if args.save_model:
            state = {name: value.detach().cpu().to(torch.bfloat16) for name, value in policy.state_dict().items()}
            policy.save_pretrained(output / "final-model", state_dict=state, safe_serialization=True)
            tokenizer.save_pretrained(output / "final-model")
        summary = {"status": "PASS", "method": args.method, "seed": args.seed,
            "profile": args.profile, "completed_rollout_steps": steps, "optimizer_updates": updates,
            "initial_validation": initial, "final_validation": final, "best_val_reward": best,
            "model_probe_changed": initial_probe != probe(), "signal_rollouts": signal_updates,
            "elapsed_seconds": time.monotonic() - started, "wandb_url": run.url,
            "checkpoint_saved": args.save_model}
        write_json(output / "completed.json", summary)
        run.summary.update(summary)
        completed = True
        print("A5_TRAINING_COMPLETE", json.dumps(summary), flush=True)
    except BaseException as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                                            "optimizer_updates": updates})
        raise
    finally:
        if sampler is not None:
            sampler.stop()
        if run is not None:
            run.finish(exit_code=0 if completed else 1)


if __name__ == "__main__":
    main()
