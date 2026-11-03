"""Train the A5 RL variants, DAPO, or full-vocabulary on-policy distillation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import signal

ROOT = Path(__file__).resolve().parent
RL_METHODS = ("grpo", "grpo_constant", "dr_grpo", "rft", "maxrl", "soft_maxrl",
              "offpolicy_naive", "offpolicy_noclip", "offpolicy_clip", "offpolicy_gspo")


def optimize_batch(policy, optimizer, batch, indices, cfg, method, settings=None,
                   advantages=None, old=None, teacher=None, full_count=None):
    """Accumulate microbatches, then perform exactly one optimizer update."""
    import torch
    from algorithms import loss_for_method, token_loss
    from distillation import response_objective
    from model_utils import response_log_probs
    device = next(policy.parameters()).device
    full_count = len(indices) if full_count is None else full_count
    denominator = int(batch["response_mask"][indices].sum())
    optimizer.zero_grad(set_to_none=True)
    policy.train()
    total_loss, statistics = 0., []
    for begin in range(0, len(indices), cfg["microbatch_size"]):
        selected = indices[begin:begin + cfg["microbatch_size"]]
        ids, labels, mask = [batch[k][selected].to(device)
                             for k in ("input_ids", "labels", "response_mask")]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if method == "opd":
                loss, info = response_objective(policy, teacher, ids, labels, mask,
                                                denominator, cfg["head_chunk_tokens"])
                statistics.append(info)
            else:
                current = response_log_probs(policy, ids, labels, mask,
                                             head_chunk_size=cfg["head_chunk_tokens"])["log_probs"]
                frozen = None if old is None else old[selected].to(device)
                a = advantages[selected].to(device)
                if method == "dapo":
                    loss, _ = token_loss(current, frozen, a, mask, denominator,
                                         cfg["clip_low"], cfg["clip_high"])
                else:
                    loss, _ = loss_for_method(current, frozen, a, mask, settings,
                                              full_count, cfg["max_new_tokens"])
                    if method.startswith("offpolicy_") or settings["loss_normalization"] == "sequence":
                        loss = loss * (len(selected) / full_count)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        total_loss += float(loss.detach())
    if not indices:
        # Preserve the original estimators' zero-signal Adam behavior.
        for parameter in policy.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
    norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"],
                                               error_if_nonfinite=True))
    if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
        raise RuntimeError("The teacher must remain frozen")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metric = {"loss": total_loss, "grad_norm": norm}
    if statistics:
        info = torch.cat(statistics)
        metric.update(reverse_kl=float(info[:, 0].mean()),
                      student_entropy=float(info[:, 1].mean()),
                      teacher_entropy=float(info[:, 2].mean()))
    return metric


def frozen_log_probs(policy, batch, cfg):
    import torch
    from model_utils import response_log_probs
    device = next(policy.parameters()).device
    result = torch.zeros(batch["labels"].shape)
    policy.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        for start in range(0, len(result), cfg["microbatch_size"]):
            end = start + cfg["microbatch_size"]
            ids, labels, mask = [batch[k][start:end].to(device)
                                 for k in ("input_ids", "labels", "response_mask")]
            result[start:end] = response_log_probs(policy, ids, labels, mask,
                head_chunk_size=cfg["head_chunk_tokens"])["log_probs"].cpu()
    return result.detach()


def collect_rollout(sampler, train, tokenizer, template, cfg, method, rng, step):
    from algorithms import select_mixed_groups
    from utils import stats
    if method != "dapo":
        examples = rng.sample(train, cfg["rollout_batch_size"] // cfg["group_size"])
        return sampler.generate(examples, tokenizer, template, cfg["group_size"],
                                cfg["seed"] + step * 1000), {}
    candidates = rng.sample(train, cfg["max_candidate_groups"])
    kept, generated = [], []
    counts = {"all_wrong": 0, "all_correct": 0, "mixed": 0}
    for wave, start in enumerate(range(0, len(candidates), cfg["generation_prompt_batch"]), 1):
        rows = sampler.generate(candidates[start:start + cfg["generation_prompt_batch"]],
                                tokenizer, template, cfg["group_size"], cfg["seed"] + step * 1000 + wave)
        accepted, c = select_mixed_groups(rows, cfg["group_size"])
        for key, value in c.items():
            counts[key] += value
        kept.extend(accepted[:cfg["rollout_batch_size"] - len(kept)])
        generated.extend(rows)
        if len(kept) == cfg["rollout_batch_size"]:
            return kept, {"candidate_reward": stats(generated)["reward"],
                          "generated_responses": len(generated), **counts}
    raise RuntimeError("Dynamic sampling exhausted the candidate budget without enough mixed groups")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=(*RL_METHODS, "dapo", "opd"), required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--teacher")
    parser.add_argument("--prompt", type=Path, default=ROOT / "prompt.prompt")
    parser.add_argument("--steps", type=int, help="Rollout iterations; OPD performs one update per iteration")
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--trainer-device", default="cuda:0")
    parser.add_argument("--sampler-gpu", type=int, default=1)
    parser.add_argument("--sampler-port", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-model", action="store_true", help="Also export best/final inference models")
    parser.add_argument("--wandb-project", help="Enable W&B with your own login")
    parser.add_argument("--wandb-entity")
    return parser.parse_args()


def main():
    args = parse_args()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from algorithms import advantages_for_method, normalized_advantages, shaped_rewards, warmup_lr
    from checkpoints import restore_checkpoint, save_checkpoint
    from model_utils import pack_prompt_and_response_ids
    from sampler import Sampler
    from utils import Metrics, data_rows, stats, write_json

    profile = args.method if args.method in ("dapo", "opd") else "rl"
    cfg = json.loads((args.config or ROOT / "configs" / (profile + ".json")).read_text())
    settings = cfg.pop("methods", {}).get(args.method)
    if args.method in RL_METHODS and settings is None:
        raise ValueError("The RL configuration is missing this method's estimator settings")
    cfg.update(method=args.method)
    if settings is not None:
        cfg["estimator"] = settings
    for key, value in (("model", args.model), ("teacher", args.teacher),
                       ("rollout_steps", args.steps), ("microbatch_size", args.microbatch_size)):
        if value is not None:
            cfg[key] = value
    cfg["prompt_sha256"] = hashlib.sha256(args.prompt.read_bytes()).hexdigest()
    for key, path in (("train", args.train_file), ("val", args.val_file)):
        cfg[key + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if min(cfg[k] for k in ("rollout_steps", "rollout_batch_size", "microbatch_size", "group_size")) < 1:
        raise ValueError("Training sizes must be positive")
    if cfg["rollout_batch_size"] % cfg["group_size"]:
        raise ValueError("Incomplete prompt groups")
    if (args.output / "config.json").exists() and args.resume is None:
        raise FileExistsError("Existing output preserved; choose another output or --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    device = torch.device(args.trainer_device)
    torch.cuda.set_device(device)
    if torch.cuda.current_device() == args.sampler_gpu:
        raise ValueError("Trainer and sampler must use different GPUs")
    train = data_rows(args.train_file, cfg["train_examples"], cfg["data_subset_seed"])
    validation = data_rows(args.val_file, cfg["val_examples"], cfg["data_subset_seed"])
    template = args.prompt.read_text()
    if any(row["question"] in template for row in train + validation):
        raise ValueError("Prompt examples overlap the selected questions")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    policy = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32,
        attn_implementation="sdpa").to(device)
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    teacher = None
    if args.method == "opd":
        revision = {} if Path(cfg["teacher"]).exists() else {"revision": cfg["teacher_revision"]}
        teacher_tokenizer = AutoTokenizer.from_pretrained(cfg["teacher"], **revision)
        if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            raise ValueError("Full-vocabulary distillation requires matching token-ID mappings")
        probes = [template.format(question=r["question"]) for r in train[:8]]
        if any(tokenizer.encode(s, add_special_tokens=False) !=
               teacher_tokenizer.encode(s, add_special_tokens=False) for s in probes):
            raise ValueError("Student and teacher tokenize the same prompt differently")
        teacher = AutoModelForCausalLM.from_pretrained(cfg["teacher"], dtype=torch.bfloat16,
            attn_implementation="sdpa", **revision).to(device).eval().requires_grad_(False)
        teacher.config.use_cache = False
        if policy.config.vocab_size != teacher.config.vocab_size:
            raise ValueError("Student and teacher output vocabularies differ")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["learning_rate"],
        betas=tuple(cfg["adam_betas"]), weight_decay=cfg["weight_decay"], foreach=False)
    progress = {"step": 0, "optimizer_updates": 0, "best_reward": -1.}
    if args.resume:
        saved = restore_checkpoint(args.resume, policy, optimizer, rng)
        if saved["config"] != cfg:
            raise ValueError("Checkpoint configuration does not match this run")
        progress = saved["progress"]
    write_json(args.output / "config.json", cfg)
    metrics = Metrics(args.output, cfg, args.wandb_project, args.wandb_entity)
    sampler = Sampler(cfg["model"], args.output, cfg, args.sampler_gpu, args.sampler_port)

    def terminate(number, frame):
        raise SystemExit(128 + number)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    def export(name):
        target = args.output / name
        weights = {key: value.detach().cpu().to(torch.bfloat16)
                   for key, value in policy.state_dict().items()}
        policy.save_pretrained(target, state_dict=weights, safe_serialization=True)
        tokenizer.save_pretrained(target)

    def evaluate(step):
        rows = sampler.generate(validation, tokenizer, template, 1, cfg["eval_seed"])
        result = stats(rows)
        metrics.log({"step": step, "optimizer_updates": progress["optimizer_updates"],
                     **{"val/" + key: value for key, value in result.items()}})
        if result["reward"] > progress["best_reward"]:
            progress["best_reward"] = result["reward"]
            if args.save_model:
                export("best-model")
        progress["last_validation"] = result

    try:
        sampler.start()
        sampler.init_weight_sync(device)
        sampler.sync_policy_weights(policy)
        if not args.resume:
            evaluate(0)
        for step in range(progress["step"] + 1, cfg["rollout_steps"] + 1):
            records, sampling = collect_rollout(sampler, train, tokenizer, template, cfg, args.method, rng, step)
            batch = pack_prompt_and_response_ids([r["prompt_token_ids"] for r in records],
                [r["response_token_ids"] for r in records], tokenizer.pad_token_id)
            if any(batch["labels"][i][batch["response_mask"][i]].tolist() != r["response_token_ids"]
                   for i, r in enumerate(records)):
                raise RuntimeError("Training labels differ from sampled actions")
            reward = torch.tensor([r["rewards"]["reward"] for r in records])
            off_policy = args.method.startswith("offpolicy_")
            old = frozen_log_probs(policy, batch, cfg) if off_policy or args.method == "dapo" else None
            advantages = None
            if args.method == "dapo":
                shaped, _ = shaped_rewards(reward, [len(r["response_token_ids"]) for r in records],
                                           cfg["max_new_tokens"], cfg["overlong_cache"])
                advantages = normalized_advantages(shaped, cfg["group_size"])
            elif args.method != "opd":
                advantages = advantages_for_method(reward, cfg["group_size"], settings)
            order = list(range(len(records)))
            if args.method != "opd":
                random.Random(cfg["seed"] + 1000000 + step).shuffle(order)
            if args.method in RL_METHODS and not off_policy and args.method != "grpo":
                order = [i for i in order if advantages[i] != 0]
            update_size = cfg["offpolicy_train_batch_size"] if off_policy else cfg["train_batch_size"]
            groups = [order[i:i + update_size] for i in range(0, len(order), update_size)] or [[]]
            measurements = []
            for indices in groups:
                update = progress["optimizer_updates"] + 1
                lr = warmup_lr(update, cfg["learning_rate"], cfg["warmup_updates"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                full_count = len(records) if args.method in RL_METHODS and not off_policy else len(indices)
                measurements.append(optimize_batch(policy, optimizer, batch, indices, cfg, args.method,
                    settings, advantages, old, teacher, full_count))
                progress["optimizer_updates"] = update
            sampler.sync_policy_weights(policy)
            row = {"step": step, "optimizer_updates": progress["optimizer_updates"],
                   **{"train/" + k: v for k, v in stats(records, cfg["group_size"]).items()},
                   **{"sampling/" + k: v for k, v in sampling.items()},
                   "train/loss": sum(r["loss"] for r in measurements),
                   "train/grad_norm": max(r["grad_norm"] for r in measurements), "train/lr": lr}
            if args.method == "opd":
                row.update({"train/" + key: measurements[0][key]
                            for key in ("reverse_kl", "student_entropy", "teacher_entropy")})
            metrics.log(row)
            progress["step"] = step
            if step % cfg["eval_every"] == 0 or step == cfg["rollout_steps"]:
                evaluate(step)
            if step == 1 or step % cfg["checkpoint_every"] == 0 or step == cfg["rollout_steps"]:
                save_checkpoint(args.output / "checkpoint.pt", policy, optimizer, rng, progress, cfg)
        if args.save_model:
            export("final-model")
        write_json(args.output / "result.json", progress)
    finally:
        sampler.stop()
        metrics.close()


if __name__ == "__main__":
    main()
