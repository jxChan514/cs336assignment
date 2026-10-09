"""Assignment 1 §7.2：batch size 实验，复用自己的模型、AdamW 和数据采样。

每张 GPU 运行这个脚本的一份进程，各自保存 checkpoint 和 W&B run。
按 token 数确定学习率和验证时机；不同 batch 使用相同训练预算和验证样本。
学习用的 train.py 和已完成的学习率实验脚本保持原样。
"""

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.experiment import ExperimentLogger
from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-learning-rate", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=327_680_000)
    parser.add_argument("--warmup-tokens", type=int, default=3_276_800)
    parser.add_argument("--total-steps", type=int, help="短流程检查时覆盖 token-budget")
    parser.add_argument("--warmup-iters", type=int, help="短流程检查时覆盖 warmup-tokens")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("float32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-tokens", type=int, default=327_680)
    parser.add_argument("--eval-every-tokens", type=int, default=8_192_000)
    parser.add_argument("--save-every-tokens", type=int, default=16_384_000)
    parser.add_argument("--log-interval", type=int, help="覆盖 log-every-tokens")
    parser.add_argument("--eval-interval", type=int, help="覆盖 eval-every-tokens")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, help="覆盖 save-every-tokens")
    parser.add_argument("--resume", type=Path, help="本脚本保存的存档包含采样随机状态，可继续训练")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-entity", default="mrxian123456-ca-si-a")
    parser.add_argument("--wandb-project", default="cs336-assignment1")
    parser.add_argument("--wandb-group", default="tinystories-batch-20261009")
    parser.add_argument("--run-name")
    parser.add_argument("--benchmark", action="store_true", help="仅测速度和显存，不创建 W&B run 或存档")
    args = parser.parse_args()
    for name in ("batch_size", "context_length", "vocab_size", "d_model", "d_ff", "num_layers", "num_heads", "eval_batch_size", "eval_batches", "token_budget", "log_every_tokens", "eval_every_tokens", "save_every_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} 必须为正数")
    tokens_per_step = args.batch_size * args.context_length
    if args.total_steps is not None:
        if args.total_steps <= 0:
            parser.error("total-steps 必须为正数")
        args.token_budget = args.total_steps * tokens_per_step
    args.total_steps = math.ceil(args.token_budget / tokens_per_step)
    if args.warmup_iters is not None:
        args.warmup_tokens = args.warmup_iters * tokens_per_step
    args.warmup_iters = math.ceil(args.warmup_tokens / tokens_per_step)
    for name in ("log", "eval", "save"):
        interval = getattr(args, f"{name}_interval")
        if interval is not None:
            if interval <= 0:
                parser.error(f"{name}-interval 必须为正数")
            setattr(args, f"{name}_every_tokens", interval * tokens_per_step)
    if args.d_model % args.num_heads or (args.d_model // args.num_heads) % 2:
        parser.error("每个头的维度必须为偶数")
    if not 0 <= args.warmup_tokens < args.token_budget or args.max_learning_rate <= 0:
        parser.error("需要 0 <= warmup-tokens < token-budget，且学习率为正数")
    if args.token_budget % args.context_length:
        parser.error("token-budget 必须是 context-length 的整数倍")
    if args.precision == "bf16" and not args.device.startswith("cuda"):
        parser.error("本脚本的 bf16 模式用于 CUDA；CPU 调试请用 float32")
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_data = np.memmap(args.data_dir / "train.bin", dtype="<u2", mode="r")
    valid_data = np.memmap(args.data_dir / "valid.bin", dtype="<u2", mode="r")
    for data in (train_data, valid_data):
        if len(data) <= args.context_length or int(data.max()) >= args.vocab_size:
            raise ValueError("数据长度或 token ID 与模型配置不匹配")
    model = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.d_ff, args.num_layers, args.num_heads, 10_000.0, device=args.device, dtype=torch.float32)
    optimizer = AdamW(model.parameters(), args.max_learning_rate, (0.9, 0.95), 1e-8, 0.1)
    parameter_count = sum(p.numel() for p in model.parameters())
    configuration = json.loads(json.dumps(vars(args), default=str))
    configuration.update({"parameters": parameter_count, "planned_tokens": args.token_budget, "model_dtype": "float32", "betas": [0.9, 0.95], "weight_decay": 0.1, "grad_clip": 1.0, "min_lr_ratio": 0.1, "schedule_unit": "tokens", "eval_tokens": args.eval_batch_size * args.eval_batches * args.context_length})
    manifest_path = args.data_dir / "manifest.json"
    if manifest_path.exists():
        configuration["dataset"] = json.loads(manifest_path.read_text())
    configuration["torch_version"] = str(torch.__version__)
    if args.device.startswith("cuda"):
        configuration["gpu"] = torch.cuda.get_device_name()
    print(json.dumps(configuration, indent=2), flush=True)
    print(f"model parameters: {parameter_count:,}", flush=True)

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()

    def synchronize():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    def evaluate():
        # 验证使用独立且固定的随机样本，不改变训练 batch 的采样状态。
        state = np.random.get_state()
        np.random.seed(args.seed + 1)
        model.eval()
        losses = []
        try:
            with torch.no_grad():
                for _ in range(args.eval_batches):
                    x, y = get_batch(valid_data, args.eval_batch_size, args.context_length, args.device)
                    with autocast():
                        logits = model(x)
                    losses.append(cross_entropy(logits.float(), y).item())
        finally:
            np.random.set_state(state)
            model.train()
        return sum(losses) / len(losses)

    start_step, tokens_seen, elapsed_offset = 0, 0, 0.0
    best_valid = math.inf
    resume = None
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        start_step, tokens_seen = resume["iteration"], resume["tokens_seen"]
        elapsed_offset = resume["elapsed_seconds"]
        best_valid = resume["best_valid_loss"]
        if tokens_seen > args.token_budget:
            raise ValueError("存档进度超过目标 token-budget")
        saved_configuration = resume["training_configuration"]
        for name in ("batch_size", "context_length", "vocab_size", "d_model", "d_ff", "num_layers", "num_heads", "precision", "seed", "max_learning_rate", "token_budget", "warmup_tokens"):
            if saved_configuration[name] != configuration[name]:
                raise ValueError(f"恢复配置不一致：{name}")
    run = None
    logger = None
    if not args.benchmark:
        import wandb
        # 指定 mode，避免服务器激活脚本默认的 offline 设置覆盖这次在线实验。
        identity = args.output_dir / "wandb_run.json"
        existing_id = json.loads(identity.read_text())["id"] if args.resume and identity.exists() else None
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group, name=args.run_name or f"batch-{args.batch_size}-lr-{args.max_learning_rate:g}", config=configuration, dir=str(args.output_dir), mode=args.wandb_mode, id=existing_id, resume="allow" if existing_id else None, settings=wandb.Settings(init_timeout=90))
        run.define_metric("step")
        run.define_metric("tokens_seen")
        run.define_metric("elapsed_seconds")
        for name in ("train/loss", "valid/loss", "lr"):
            run.define_metric(name, step_metric="tokens_seen")
        for name in ("steps/train_loss", "steps/valid_loss"):
            run.define_metric(name, step_metric="step")
        for name in ("time/train_loss", "time/valid_loss"):
            run.define_metric(name, step_metric="elapsed_seconds")
        run.define_metric("valid/loss", step_metric="tokens_seen", summary="min")
        (args.output_dir / "wandb_run.json").write_text(json.dumps({"id": run.id, "url": run.url, "entity": run.entity, "project": run.project}, indent=2), encoding="utf-8")
        logger = ExperimentLogger(args.output_dir, {"arguments": configuration, "model_dtype": "torch.float32", "start_step": start_step}, start_step=start_step)
        print(f"W&B run: {run.url}", flush=True)
    np.random.seed(args.seed)
    if resume is not None:
        random_state = resume["numpy_rng"]
        np.random.set_state((random_state[0], np.asarray(random_state[1], dtype=np.uint32), random_state[2], random_state[3], random_state[4]))
        torch.set_rng_state(resume["torch_rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state(resume["cuda_rng"])
    model.train()
    final_valid = None
    completed = start_step
    status = "running"
    reason = None
    excessive_losses = 0
    measured_started = None
    training_peak_gib = 0.0
    validation_peak_gib = 0.0
    reserved_peak_gib = 0.0
    synchronize()
    started = time.perf_counter()
    next_log = (tokens_seen // args.log_every_tokens + 1) * args.log_every_tokens
    next_eval = (tokens_seen // args.eval_every_tokens + 1) * args.eval_every_tokens
    next_save = (tokens_seen // args.save_every_tokens + 1) * args.save_every_tokens

    def save_training_checkpoint(path):
        """长时间的 batch=1 实验也保存采样进度，临时文件写完再替换存档。"""
        rng = np.random.get_state()
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "iteration": completed, "tokens_seen": tokens_seen,
            "elapsed_seconds": elapsed_offset + time.perf_counter() - started,
            "best_valid_loss": best_valid,
            "numpy_rng": (rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if args.device.startswith("cuda") else None,
            "training_configuration": configuration,
        }
        temporary = path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    try:
        for step in range(start_step, args.total_steps):
            lr = get_lr_cosine_schedule(tokens_seen, args.max_learning_rate, args.max_learning_rate * 0.1, args.warmup_tokens, args.token_budget)
            # 最后一步可以用较小 batch，确保每组恰好处理相同数量的 token。
            current_batch = min(args.batch_size, (args.token_budget - tokens_seen) // args.context_length)
            x, y = get_batch(train_data, current_batch, args.context_length, args.device)
            optimizer.zero_grad()
            with autocast():
                logits = model(x)
            # FP32 权重和优化器状态；交叉熵的 exp/sum 也使用 FP32。
            loss = cross_entropy(logits.float(), y)
            train_loss = loss.item()
            excessive_losses = excessive_losses + 1 if train_loss > 100 else 0
            if not math.isfinite(train_loss) or excessive_losses >= 10:
                status = "diverged"
                reason = "nonfinite_loss" if not math.isfinite(train_loss) else "loss_above_100_for_10_steps"
                if logger:
                    logger.log(step + 1, train_loss, None, lr)
                if run:
                    run.log({"step": step + 1, "train/loss": train_loss, "lr": lr})
                break
            loss.backward()
            gradient_clipping(model.parameters(), 1.0)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            completed = step + 1
            tokens_seen += current_batch * args.context_length
            valid_loss = None
            if args.benchmark and completed == 3:
                synchronize()
                measured_started = time.perf_counter()
            if not args.benchmark and (tokens_seen >= next_eval or completed == args.total_steps):
                while next_eval <= tokens_seen:
                    next_eval += args.eval_every_tokens
                if args.device.startswith("cuda"):
                    training_peak_gib = max(training_peak_gib, torch.cuda.max_memory_allocated() / 1024**3)
                    reserved_peak_gib = max(reserved_peak_gib, torch.cuda.max_memory_reserved() / 1024**3)
                    torch.cuda.reset_peak_memory_stats()
                valid_loss = evaluate()
                if args.device.startswith("cuda"):
                    validation_peak_gib = max(validation_peak_gib, torch.cuda.max_memory_allocated() / 1024**3)
                    reserved_peak_gib = max(reserved_peak_gib, torch.cuda.max_memory_reserved() / 1024**3)
                    torch.cuda.reset_peak_memory_stats()
                final_valid = valid_loss
                if not math.isfinite(valid_loss):
                    status, reason = "diverged", "nonfinite_validation_loss"
                elif valid_loss < best_valid:
                    best_valid = valid_loss
                    save_training_checkpoint(args.output_dir / "best.pt")
            if tokens_seen >= next_log or valid_loss is not None or completed == args.total_steps:
                while next_log <= tokens_seen:
                    next_log += args.log_every_tokens
                elapsed = elapsed_offset + time.perf_counter() - started
                print(f"step={completed} tokens={tokens_seen:,} train={train_loss:.5f} valid={valid_loss} lr={lr:.6g} seconds={elapsed:.1f}", flush=True)
                if logger:
                    logger.log(completed, train_loss, valid_loss, lr)
                if run:
                    values = {"step": completed, "elapsed_seconds": elapsed, "train/loss": train_loss, "steps/train_loss": train_loss, "time/train_loss": train_loss, "lr": lr, "tokens_seen": tokens_seen}
                    if valid_loss is not None:
                        values.update({"valid/loss": valid_loss, "steps/valid_loss": valid_loss, "time/valid_loss": valid_loss})
                    run.log(values)
            if not args.benchmark and tokens_seen >= next_save:
                while next_save <= tokens_seen:
                    next_save += args.save_every_tokens
                save_training_checkpoint(args.output_dir / "latest.pt")
            if status == "diverged":
                break
        else:
            status = "completed"
        synchronize()
        elapsed = elapsed_offset + time.perf_counter() - started
        if args.device.startswith("cuda"):
            training_peak_gib = max(training_peak_gib, torch.cuda.max_memory_allocated() / 1024**3)
            reserved_peak_gib = max(reserved_peak_gib, torch.cuda.max_memory_reserved() / 1024**3)
        peak = max(training_peak_gib, validation_peak_gib) if args.device.startswith("cuda") else None
        summary = {"status": status, "reason": reason, "completed_steps": completed, "tokens_seen": tokens_seen, "elapsed_seconds": elapsed, "best_valid_loss": best_valid if math.isfinite(best_valid) else None, "final_valid_loss": final_valid, "peak_memory_gib": peak, "training_peak_memory_gib": training_peak_gib, "validation_peak_memory_gib": validation_peak_gib, "peak_reserved_memory_gib": reserved_peak_gib, "tokens_per_second": tokens_seen / elapsed}
        if args.benchmark:
            summary["seconds_per_step_after_warmup"] = (time.perf_counter() - measured_started) / (completed - 3) if measured_started and completed > 3 else None
        else:
            if status == "completed":
                save_training_checkpoint(args.output_dir / "final.pt")
            if run:
                run.summary.update(summary)
            from cs336_basics.plot_metrics import plot_metrics
            plot_metrics(logger.metrics_path)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        if run:
            run.summary.update({"status": "failed", "error_type": type(error).__name__, "completed_steps": completed})
        raise
    finally:
        if run:
            run.finish(exit_code=0 if status in ("completed", "diverged") else 1)


if __name__ == "__main__":
    main()
