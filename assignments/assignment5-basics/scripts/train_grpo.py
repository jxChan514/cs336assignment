"""On-policy A5 RL variants, with separate or alternating GPUs and W&B examples."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import hashlib
import html
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
METHODS = {
    "grpo": ("mean", "std", "sequence"),
    "grpo_constant": ("mean", "std", "constant"),
    "dr_grpo": ("mean", "none", "constant"),
    "rft": ("none", "none", "constant"),
    "maxrl": ("mean", "mean", "constant"),
}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_gsm8k(path: Path, count: int):
    rows = []
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if "####" not in row["answer"]:
            raise ValueError(f"GSM8K row {index} has no #### answer delimiter")
        rows.append({"id": index, "question": row["question"],
                     "ground_truth": row["answer"].rsplit("####", 1)[1].strip()})
    if not 0 < count <= len(rows):
        raise ValueError(f"Requested {count} examples, but {path} contains {len(rows)}")
    # Same data subsets and fixed comparison questions across all experiment seeds.
    random.Random(42).shuffle(rows)
    return rows[:count]


def summarize_records(records, group_size=1):
    import numpy as np
    rewards = np.asarray([r["rewards"]["reward"] for r in records], dtype=float)
    formats = np.asarray([r["rewards"]["format_reward"] for r in records], dtype=float)
    lengths = np.asarray([r["response_tokens"] for r in records], dtype=float)
    groups = rewards.reshape(-1, group_size)
    stats = {"reward": float(rewards.mean()), "format_reward": float(formats.mean()),
             "response_length": float(lengths.mean()),
             "truncated_fraction": sum(r["finish_reason"] == "length" for r in records) / len(records)}
    if group_size > 1:
        stds = groups.std(axis=1, ddof=1)
        stats.update(group_std_mean=float(stds.mean()), zero_std_group_fraction=float((stds == 0).mean()),
                     pass_at_group=float((groups.max(axis=1) > 0).mean()))
    return stats


def comparison_html(before, after, step):
    """Escaped, side-by-side cards for the same questions (no generated HTML execution)."""
    initial = {r["id"]: r for r in before}
    cards = []
    for row in after:
        old = initial[row["id"]]
        def cell(record, heading):
            reward = record["rewards"]["reward"]
            color = "#166534" if reward else "#991b1b"
            return (f'<section><h3>{heading}</h3><p style="color:{color}">'
                    f'奖励 {reward:g} · 格式 {record["rewards"]["format_reward"]:g} · '
                    f'{record["response_tokens"]} tokens</p>'
                    f'<pre>{html.escape(record["response"])}</pre></section>')
        cards.append(f'<article><h2>题目 #{row["id"]}</h2><p>{html.escape(row["question"])}</p>'
                     f'<p>标准答案：{html.escape(row["ground_truth"])}</p><div class="pair">'
                     + cell(old, "训练前 · step 0") + cell(row, f"更新后 · step {step}") + '</div></article>')
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GRPO 回答对比</title>'
            '<style>body{font:16px system-ui;max-width:1400px;margin:24px auto;padding:16px;'
            'background:#f6f7fb;color:#172033}article{background:white;padding:24px;margin:20px 0;'
            'border-radius:12px}h2{font-size:18px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}'
            'section{min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 monospace;'
            'padding:16px;background:#f1f5f9;border-radius:8px}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style>'
            '<h1>固定验证题：训练前后回答对比</h1>'
            '<p>颜色表示作业评分器的奖励；同一题保持相同评测采样 seed。单次回答不能代表整体正确率。</p>'
            + "".join(cards) + '</html>')


def balanced_examples(records, count):
    buckets = [[r for r in records if r["rewards"]["reward"] > 0],
               [r for r in records if r["rewards"]["reward"] == 0 and r["rewards"]["format_reward"] > 0],
               [r for r in records if r["rewards"]["format_reward"] == 0]]
    selected = []
    while len(selected) < count and any(buckets):
        for bucket in buckets:
            if bucket and len(selected) < count:
                selected.append(bucket.pop(0))
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "pilot", "standard"], default="pilot")
    parser.add_argument("--method", choices=list(METHODS), default="grpo")
    parser.add_argument("--model", default=os.environ.get("CS336_A5_MODEL_DIR", str(PROJECT / "models/OLMo-2-0425-1B")))
    data = Path(os.environ.get("CS336_A5_DATA_DIR", str(PROJECT / "data/gsm8k")))
    parser.add_argument("--train-file", type=Path, default=data / "train.jsonl")
    parser.add_argument("--val-file", type=Path, default=data / "test.jsonl")
    parser.add_argument("--prompt", choices=["r1_zero", "question_only", "r1_zero_three_shot_gsm8k"], default="r1_zero")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--train-examples", type=int)
    parser.add_argument("--val-examples", type=int)
    parser.add_argument("--rollout-batch-size", type=int)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=1, help="Responses per training GPU per microbatch")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--attention-implementation", choices=["flash_attention_2", "sdpa"], default="flash_attention_2")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--samples-every", type=int, default=40)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--sampling-gpus", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--execution-mode", choices=["separate_gpus", "single_gpu"], default="separate_gpus")
    parser.add_argument("--port", type=int, default=18120)
    parser.add_argument("--vllm-memory-utilization", type=float, default=0.75)
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument("--request-batch-size", type=int, default=64)
    parser.add_argument("--wandb-project", default="cs336-a5-grpo")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-group", default="standard-on-policy-3090")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-save-model", action="store_true")
    args = parser.parse_args()
    defaults = {"smoke": (2, 128, 32, 32), "pilot": (50, 6400, 1024, 256), "standard": (200, 6400, 1024, 256)}
    for key, value in zip(("steps", "train_examples", "val_examples", "rollout_batch_size"), defaults[args.profile]):
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.profile == "smoke":
        args.eval_every = 1
    positive = ("steps", "train_examples", "val_examples", "rollout_batch_size", "microbatch_size", "eval_every",
                "samples_every", "sample_count", "request_batch_size", "max_tokens", "max_grad_norm", "learning_rate")
    if any(getattr(args, key) <= 0 for key in positive) or args.group_size < 2 or args.temperature < 0:
        parser.error("Counts and training scales must be positive; group-size must be >= 2")
    if not 0 < args.vllm_memory_utilization < 1:
        parser.error("vllm-memory-utilization must be between 0 and 1")
    if len(set(args.sampling_gpus)) != len(args.sampling_gpus) or not args.sampling_gpus:
        parser.error("sampling-gpus must contain distinct GPU indices")
    return args


def main():
    import torch
    import torch.distributed as dist
    from torch.distributed.optim import ZeroRedundancyOptimizer
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import wandb
    from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
    from cs336_alignment.grpo import grpo_train_step
    from cs336_alignment.vllm_utils import VLLMServer, wait_for_server, _http_json

    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        control_group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
    else:
        control_group = None
    visible = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")]
    single_gpu = args.execution_mode == "single_gpu"
    if single_gpu and (world_size != 1 or args.sampling_gpus != visible[:1]):
        raise ValueError("single_gpu requires one training process and the same sampling GPU")
    if not single_gpu and set(visible[:world_size]) & set(args.sampling_gpus):
        raise ValueError("Training GPUs and sampling GPUs must be disjoint")
    divisor = world_size * args.group_size
    if args.rollout_batch_size % divisor:
        raise ValueError("rollout-batch-size must be divisible by training GPU count * group-size")
    local_batch_size = args.rollout_batch_size // world_size
    if local_batch_size % args.microbatch_size:
        raise ValueError("Per-GPU batch must be divisible by microbatch-size")
    accumulation = local_batch_size // args.microbatch_size
    if args.train_examples < args.rollout_batch_size // args.group_size:
        raise ValueError("Need at least one batch of distinct training questions")

    def leader_call(fn):
        """Communicate leader-side setup/HTTP errors instead of stranding other ranks."""
        result, status = None, [None]
        if rank == 0:
            try:
                result = fn()
            except Exception as error:
                status[0] = f"{type(error).__name__}: {error}"
                logging.exception("Rank-0 operation failed")
        if world_size > 1:
            dist.broadcast_object_list(status, src=0, group=control_group)
        if status[0]:
            raise RuntimeError(status[0])
        return result

    def broadcast(value):
        container = [value]
        if world_size > 1:
            dist.broadcast_object_list(container, src=0, group=control_group)
        return container[0]

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_data = load_gsm8k(args.train_file, args.train_examples)
    val_data = load_gsm8k(args.val_file, args.val_examples)
    template = (PROJECT / "cs336_alignment/prompts" / f"{args.prompt}.prompt").read_text()
    grader = question_only_reward_fn if args.prompt == "question_only" else r1_zero_reward_fn
    # Format errors keep the original assignment reward of zero.
    rng = random.Random(args.seed)
    output = args.output_dir
    if rank == 0 and output is None:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
        output = PROJECT / "outputs" / f"a5-grpo-{args.profile}-{stamp}-seed{args.seed}"
    output = Path(broadcast(str(output) if rank == 0 else None)).resolve()
    servers = []
    run = None
    completed = False
    initial_samples = []
    best_reward = float("-inf")

    def terminate(signum, frame):
        for server in servers:
            server.stop()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    class ManagedServer(VLLMServer):
        def start(self):
            # Fail on an occupied port; never terminate a server belonging to another task.
            with socket.socket() as check:
                check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                check.bind((self.host, self.port))
            env = os.environ.copy()
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(key, None)
            for key in list(env):
                if key.startswith("TORCHELASTIC_"):
                    env.pop(key)
            env.update(CUDA_VISIBLE_DEVICES=str(self.gpu), VLLM_SERVER_DEV_MODE="1", VLLM_LOGGING_LEVEL="INFO")
            command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", self.model_id,
                       "--host", self.host, "--port", str(self.port), "--dtype", "bfloat16",
                       "--enable-prefix-caching", "--gpu-memory-utilization", str(self.gpu_memory_utilization),
                       "--seed", str(self.seed), "--tensor-parallel-size", "1",
                       "--max-model-len", str(args.vllm_max_model_len),
                       "--max-num-seqs", "128", "--enforce-eager"]
            if single_gpu:
                command.extend(["--max-num-batched-tokens", "2048"])
            else:
                command.extend(["--weight-transfer-config", json.dumps({"backend": "nccl"})])
            log = (output / f"vllm-gpu{self.gpu}.log").open("a")
            try:
                self.process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            finally:
                log.close()
            wait_for_server(self.base_url, self.process, self.startup_timeout)

    def setup_logging():
        nonlocal run
        if (output / "config.json").exists():
            raise FileExistsError(f"Output already exists: {output}. Choose a new output directory.")
        output.mkdir(parents=True, exist_ok=True)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(training_gpus=visible[:world_size], training_world_size=world_size,
                      gradient_accumulation_steps_per_gpu=accumulation,
                      effective_microbatch_size=world_size * args.microbatch_size,
                      precision="FP32 master weights and optimizer states; BF16 autocast and sampling",
                      optimizer="AdamW", betas=[0.9, 0.95], weight_decay=0.0,
                      ddp_bucket_warmup_passes=2 if world_size > 1 else 0,
                      weight_transfer=("BF16 checkpoint; stop vLLM before training" if single_gpu
                                       else "NCCL, one BF16 tensor at a time"),
                      train_batch_size=args.rollout_batch_size, top_p=1.0,
                      baseline=METHODS[args.method][0], advantage_normalizer=METHODS[args.method][1], importance_reweighting_method="none",
                      loss_normalization=METHODS[args.method][2], validation_seed=100000 + args.seed,
                      trajectory_token_source="vllm_token_ids", prompt_input_type="token_ids",
                      retokenize_sampled_responses=False,
                      normalization_constant_local_batch=(local_batch_size * args.max_tokens
                                                         if METHODS[args.method][2] == "constant" else None),
                      normalization_constant_global=(args.rollout_batch_size * args.max_tokens
                                                     if METHODS[args.method][2] == "constant" else None),
                      entropy_scope=("nonzero_advantage_responses" if METHODS[args.method][2] == "constant" and world_size == 1
                                     else "all_responses"),
                      validation_ids=[r["id"] for r in val_data],
                      prompt_sha256=hashlib.sha256(template.encode()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      grpo_sha256=hashlib.sha256((PROJECT / "cs336_alignment/grpo.py").read_bytes()).hexdigest(),
                      training_gpu={"name": torch.cuda.get_device_name(device),
                                    "memory_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
                                    "capability": list(torch.cuda.get_device_capability(device))},
                      cuda_runtime=torch.version.cuda, python=sys.version,
                      versions={p: importlib.metadata.version(p) for p in ["torch", "transformers", "vllm", "wandb"]})
        for label, path in [("train", args.train_file), ("val", args.val_file)]:
            config[f"{label}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(output / "config.json", config)
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
                         group=args.wandb_group, name=f"{args.method}-{args.profile}-{args.prompt}-seed{args.seed}-{output.name}",
                         config=config, dir=str(output))
        run.define_metric("rollout_step")
        for namespace in ["train", "val", "system", "debug"]:
            run.define_metric(f"{namespace}/*", step_metric="rollout_step")
        run.summary["formal_assignment_run"] = args.profile == "standard" and args.val_examples >= 1024
        write_json(output / "wandb-run.json", {"id": run.id, "url": run.url, "mode": args.wandb_mode})
        print(f"W&B: {run.url}\nOutput: {output}\nResponses={args.rollout_batch_size}; per-GPU microbatch={args.microbatch_size}; accumulation={accumulation}", flush=True)

    def setup_servers():
        for index, gpu in enumerate(args.sampling_gpus):
            sampling_model = str(output / ".sampling-model") if single_gpu else args.model
            servers.append(ManagedServer(model_id=sampling_model, gpu=gpu, port=args.port + index,
                                         seed=args.seed + index, gpu_memory_utilization=args.vllm_memory_utilization,
                                         startup_timeout=600, shutdown_timeout=30 if single_gpu else 5))
        if single_gpu:
            return
        with ThreadPoolExecutor(max_workers=len(servers)) as executor:
            list(executor.map(lambda s: s.start(), servers))
        for server in servers:
            server.init_weight_sync(str(device))

    def sync_weights():
        if single_gpu:
            for server in servers:
                server.stop()
            checkpoint = output / ".sampling-model"
            # Generation uses current weights. No same-GPU NCCL communicator and
            # no GPU-sized BF16 shadow: conversion happens one CPU tensor at a time.
            state = {name: tensor.detach().cpu().to(torch.bfloat16)
                     for name, tensor in policy.state_dict().items()}
            policy.save_pretrained(checkpoint, state_dict=state, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            return
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
        parameters = list(policy.named_parameters())
        update_info = {"names": [name for name, _ in parameters],
                       "dtype_names": ["bfloat16"] * len(parameters),
                       "shapes": [list(param.shape) for _, param in parameters], "packed": False}
        for server in servers:
            # Cast and send one tensor at a time. A full BF16 copy plus the default
            # packed buffers would exceed 24 GiB after AdamW states are allocated.
            _http_json("POST", f"{server.base_url}/pause", timeout=60)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_http_json, "POST", f"{server.base_url}/update_weights",
                                         {"update_info": update_info}, 120)
                try:
                    with torch.cuda.stream(training_stream), torch.no_grad():
                        iterator = ((name, param.detach().to(torch.bfloat16)) for name, param in parameters)
                        NCCLWeightTransferEngine.trainer_send_weights(
                            iterator, NCCLTrainerSendWeightsArgs(group=server.weight_sync_group,
                                                                packed=False, stream=training_stream))
                        training_stream.synchronize()
                    future.result()
                except BaseException:
                    # Close the receiver connection before joining a pending HTTP thread.
                    server.stop()
                    raise
            _http_json("POST", f"{server.base_url}/reset_prefix_cache", timeout=60)
            _http_json("POST", f"{server.base_url}/resume", timeout=60)

    def generate(examples, group_size, seed):
        if single_gpu:
            torch.cuda.empty_cache()
            servers[0].start()
        # Preserve prompt-major order: question 0's G answers, then question 1's G answers.
        def worker(index):
            start = index * len(examples) // len(servers)
            end = (index + 1) * len(examples) // len(servers)
            if start == end:
                return []
            subset = examples[start:end]
            prompts = [template.format(question=e["question"]) for e in subset]
            prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
            params = {"temperature": args.temperature, "max_tokens": args.max_tokens, "n": group_size,
                      "seed": seed + index}
            if args.prompt != "question_only":
                params.update(stop=["</answer>"], include_stop_str_in_output=True)
            completions = servers[index].generate_completions(prompt_ids, params, batch_size=args.request_batch_size)
            if len(completions) != len(subset) * group_size:
                raise RuntimeError("vLLM returned the wrong number of completions")
            result = []
            for example_index, example in enumerate(subset):
                for sample_index in range(group_size):
                    completion = completions[example_index * group_size + sample_index]
                    if not completion.token_ids:
                        raise RuntimeError("Missing generated token IDs; response lengths would be invalid")
                    result.append({**example, "sample_index": sample_index,
                                   "prompt": prompts[example_index], "response": completion.text,
                                   "prompt_token_ids": prompt_ids[example_index],
                                   "response_token_ids": completion.token_ids,
                                   "response_tokens": len(completion.token_ids), "finish_reason": completion.finish_reason,
                                   "rewards": grader(completion.text, example["ground_truth"])})
            return result
        try:
            with ThreadPoolExecutor(max_workers=len(servers)) as executor:
                batches = list(executor.map(worker, range(len(servers))))
        finally:
            if single_gpu:
                servers[0].stop()
                torch.cuda.empty_cache()
        return [record for batch in batches for record in batch]

    def log_table(key, records, step):
        columns = ["step", "question_id", "sample_index", "question", "ground_truth", "response", "reward", "format_reward", "tokens", "finish_reason"]
        data = [[step, r["id"], r["sample_index"], r["question"], r["ground_truth"], r["response"],
                 r["rewards"]["reward"], r["rewards"]["format_reward"], r["response_tokens"], r["finish_reason"]] for r in records]
        run.log({"rollout_step": step, key: wandb.Table(columns=columns, data=data)})

    def evaluate(step):
        nonlocal initial_samples, best_reward
        started = time.monotonic()
        records = generate(val_data, 1, seed=100000 + args.seed)
        stats = summarize_records(records)
        payload = {f"val/{key}": value for key, value in stats.items()}
        payload.update(rollout_step=step, **{"system/eval_seconds": time.monotonic() - started})
        run.log(payload)
        best_reward = max(best_reward, stats["reward"])
        run.summary.update({"best_val_reward": best_reward, "latest_val_reward": stats["reward"]})
        write_json(output / "validation" / f"step-{step:04d}.json", records)
        samples = records[:args.sample_count]
        if step == 0:
            initial_samples = samples
        log_table("samples/fixed_validation", samples, step)
        log_table("samples/validation_categories", balanced_examples(records, args.sample_count), step)
        page = comparison_html(initial_samples, samples, step)
        (output / "samples").mkdir(exist_ok=True)
        (output / "samples" / f"comparison-step-{step:04d}.html").write_text(page)
        (output / "samples" / "latest-comparison.html").write_text(page)
        run.log({"rollout_step": step, "samples/before_after": wandb.Html(page)})
        with (output / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(payload) + "\n")
        print(f"step={step} validation reward={stats['reward']:.4f}, format={stats['format_reward']:.4f}, tokens={stats['response_length']:.1f}", flush=True)

    try:
        leader_call(setup_logging)
        # Load BF16 once to bound transient CPU memory, then retain FP32 training masters.
        policy = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                     attn_implementation=args.attention_implementation, low_cpu_mem_usage=True)
        policy = policy.float().to(device)
        policy.config.use_cache = False
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if world_size > 1:
            model = DDP(policy, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
            optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW,
                                               lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)
        else:
            model = policy
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)
        training_stream = torch.cuda.current_stream(device)
        if world_size > 1:
            # DDP rebuilds its buckets after the first backward. Do that before
            # AdamW allocates states, otherwise old + new buckets can exceed 24 GiB.
            # Zero loss, no optimizer.step(): neither pass changes model parameters.
            warmup_ids = tokenizer("DDP warmup", add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=False)
                with torch.cuda.stream(training_stream), torch.autocast("cuda", dtype=torch.bfloat16):
                    warmup_loss = model(warmup_ids).logits[..., 0].float().mean() * 0.0
                warmup_loss.backward()
                del warmup_loss
            optimizer.zero_grad(set_to_none=False)
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        leader_call(setup_servers)
        leader_call(sync_weights)
        leader_call(lambda: evaluate(0))
        for step in range(1, args.steps + 1):
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            records = None
            if rank == 0:
                chosen = rng.sample(train_data, args.rollout_batch_size // args.group_size)
            sync_started = time.monotonic()
            # The previous evaluation, if any, already synchronized these same weights.
            if step > 1 and (step - 1) % args.eval_every != 0:
                leader_call(sync_weights)
            sync_seconds = time.monotonic() - sync_started
            rollout_started = time.monotonic()
            records = leader_call(lambda: generate(chosen, args.group_size, args.seed + step * 100))
            records = broadcast(records)
            rollout_seconds = time.monotonic() - rollout_started
            local_records = records[rank * local_batch_size:(rank + 1) * local_batch_size]
            scores = {(r["response"], r["ground_truth"]): r["rewards"] for r in local_records}
            probe = policy.get_output_embeddings().weight.detach().flatten()[:4096].clone()
            train_started = time.monotonic()
            with torch.cuda.stream(training_stream), torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metadata = grpo_train_step(
                    model, tokenizer, optimizer, accumulation, args.max_grad_norm,
                    lambda response, truth: scores[(response, truth)],
                    [r["prompt"] for r in local_records], [r["response"] for r in local_records],
                    [r["ground_truth"] for r in local_records], args.group_size,
                    baseline=METHODS[args.method][0], advantage_normalizer=METHODS[args.method][1],
                    loss_normalization=METHODS[args.method][2],
                    normalization_constant=(local_batch_size * args.max_tokens
                                            if METHODS[args.method][2] == "constant" else None),
                    prompt_token_ids=[r["prompt_token_ids"] for r in local_records],
                    response_token_ids=[r["response_token_ids"] for r in local_records])
            torch.cuda.synchronize(device)
            update_norm = (policy.get_output_embeddings().weight.detach().flatten()[:4096] - probe).norm()
            entropy_count = metadata["response_token_count"].float()
            aggregate = torch.stack([loss, metadata["grad_norm"], metadata["token_entropy"] * entropy_count, entropy_count,
                                     update_norm, torch.tensor(torch.cuda.max_memory_allocated(device) / 2**30, device=device)])
            if world_size > 1:
                dist.all_reduce(aggregate)
            values = aggregate.tolist()
            if rank == 0:
                train_stats = summarize_records(records, args.group_size)
                payload = {f"train/{key}": value for key, value in train_stats.items()}
                payload.update({"rollout_step": step, "train/loss": values[0] / world_size,
                                "train/active_sequence_fraction": metadata["active_sequence_fraction"],
                                "train/backward_microbatches": metadata["backward_microbatches"],
                                "debug/sampled_token_ids_used": int(metadata["sampled_token_ids_used"]),
                                "debug/trajectory_response_token_count": metadata["trajectory_response_token_count"],
                                "train/grad_norm": values[1] / world_size, "train/token_entropy": values[2] / max(values[3], 1),
                                "debug/output_weight_update_norm": values[4] / world_size,
                                "system/mean_training_peak_gib": values[5] / world_size,
                                "system/sync_seconds": sync_seconds, "system/rollout_seconds": rollout_seconds,
                                "system/train_seconds": time.monotonic() - train_started,
                                "system/step_seconds": time.monotonic() - started})
                run.log(payload)
                with (output / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(payload) + "\n")
                print(f"step={step}/{args.steps} loss={payload['train/loss']:.5f}, reward={train_stats['reward']:.4f}, "
                      f"zero-std={train_stats['zero_std_group_fraction']:.2f}, peak={payload['system/mean_training_peak_gib']:.2f} GiB, "
                      f"seconds={payload['system/step_seconds']:.1f}", flush=True)
            def save_rollouts():
                write_json(output / "rollouts" / f"step-{step:04d}.json", records)
                log_table("samples/train", balanced_examples(records, args.sample_count), step)
            if step == 1 or step % args.samples_every == 0 or step == args.steps:
                leader_call(save_rollouts)
            if step % args.eval_every == 0 or step == args.steps:
                leader_call(sync_weights)
                leader_call(lambda: evaluate(step))

        def finish():
            if not args.no_save_model:
                # Compact inference checkpoint; exact optimizer-state resume is not implemented.
                checkpoint = output / "final-model"
                state = {name: tensor.detach().cpu().to(torch.bfloat16) for name, tensor in policy.state_dict().items()}
                policy.save_pretrained(checkpoint, state_dict=state, safe_serialization=True)
                tokenizer.save_pretrained(checkpoint)
            run.summary.update({"completed_rollout_steps": args.steps, "final_val_reward": run.summary["latest_val_reward"]})
            write_json(output / "completed.json", {"steps": args.steps, "best_val_reward": best_reward,
                                                   "final_val_reward": run.summary["latest_val_reward"], "wandb_url": run.url})
            if single_gpu:
                import shutil
                shutil.rmtree(output / ".sampling-model")
        leader_call(finish)
        completed = True
    finally:
        try:
            if rank == 0:
                for server in servers:
                    server.stop()
                if run is not None:
                    run.finish(exit_code=0 if completed else 1)
        finally:
            if world_size > 1 and dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
