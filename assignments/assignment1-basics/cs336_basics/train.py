"""Assignment 1 §5.3：训练、验证，以及 checkpoint 的保存和恢复。"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

# 支持直接运行这个文件；从包导入时不会启动训练。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping

# ---------- 1. 文件路径 ----------
# 根据当前文件的位置确定作业目录，不受终端工作目录影响。
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"

# 这里约定为原始二进制 .bin 文件，读取时使用 np.memmap。
train_data_path = data_dir / "train.bin"
valid_data_path = data_dir / "valid.bin"
data_dtype = np.uint16  # 必须与保存 token ID 时的类型一致；不是模型权重的类型。

output_dir = project_root / "checkpoints" / "debug"
checkpoint_path = output_dir / "latest.pt"
final_checkpoint_path = output_dir / "final.pt"
resume_checkpoint_path: Path | None = None  # None 表示从头训练；恢复时填写存档路径。


# ---------- 2. 模型配置 ----------
# 以下是本地调试的小模型配置，方便先检查训练流程。
vocab_size = 765  # 当前样例 tokenizer 的实际词表大小；更换数据时需对应修改。
context_length = 32  # T：每条输入序列的 token 数，也是 RoPE 的缓存容量。
d_model = 64  # D：每个 token 的隐藏向量维度。
d_ff = 192  # F：SwiGLU 的中间维度。
num_layers = 2  # TransformerBlock 的数量。
num_heads = 4  # H：注意力头数；d_model 必须能被 num_heads 整除。
rope_theta = 10_000.0  # RoPE 计算旋转频率时使用的常数。


# ---------- 3. 训练与 AdamW 配置 ----------
device = "cpu"  # 先在本地 CPU 调试；使用服务器 GPU 时改为 "cuda"。
dtype = torch.float32  # 模型权重使用浮点数；get_batch 返回的 token ID 是 torch.long。
seed = 42  # 在创建模型和采样 batch 前设置 NumPy 和 PyTorch 的随机种子。

batch_size = 2  # B：一次参数更新使用的序列数量；inputs、targets 的形状为 (B,T)。
total_steps = 200  # 总共进行多少次 optimizer.step()。

max_learning_rate = 3e-4  # 调试用的初始设置；正式实验再比较不同学习率。
min_learning_rate = 3e-5
warmup_iters = 10  # 前 10 步让学习率从 0 线性增加到最大值。
cosine_cycle_iters: int | None = None  # None 表示使用总步数；恢复时可指定原来的衰减终点。

betas = (0.9, 0.95)  # AdamW 中梯度均值 m、梯度平方均值 v 的衰减系数。
eps = 1e-8  # AdamW 更新公式中加在分母上的小常数。
weight_decay = 0.1  # AdamW 的权重衰减系数。
max_l2_norm = 1.0  # 所有参数梯度合在一起的 L2 范数超过该值时进行裁剪。


# ---------- 4. 打印、验证与保存 ----------
log_interval = 10  # 每完成 10 次更新，打印训练 loss 和当前学习率。
eval_interval = 50  # 每完成 50 次更新，在验证集上计算 loss。
eval_batches = 10  # 每次验证取 10 个 batch，并计算它们的平均 loss。
save_interval = 50  # 每完成 50 次更新保存 latest.pt；训练结束再保存 final.pt。


def parse_args() -> argparse.Namespace:
    """命令行参数覆盖上面的默认配置，方便调试和在服务器上训练。"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-data", type=Path, default=train_data_path)
    parser.add_argument("--valid-data", type=Path, default=valid_data_path)
    parser.add_argument("--data-dtype", choices=("uint16", "uint32"), default=np.dtype(data_dtype).name)
    parser.add_argument("--output-dir", type=Path, help="覆盖默认 checkpoint 保存目录")
    parser.add_argument("--resume", type=Path, default=resume_checkpoint_path, help="从指定 checkpoint 恢复")
    parser.add_argument("--device", default=device)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--vocab-size", type=int, default=vocab_size)
    parser.add_argument("--context-length", type=int, default=context_length)
    parser.add_argument("--d-model", type=int, default=d_model)
    parser.add_argument("--d-ff", type=int, default=d_ff)
    parser.add_argument("--num-layers", type=int, default=num_layers)
    parser.add_argument("--num-heads", type=int, default=num_heads)
    parser.add_argument("--rope-theta", type=float, default=rope_theta)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--total-steps", type=int, default=total_steps, help="目标总步数，包含恢复前的步数")
    parser.add_argument("--max-learning-rate", type=float, default=max_learning_rate)
    parser.add_argument("--min-learning-rate", type=float, default=min_learning_rate)
    parser.add_argument("--warmup-iters", type=int, default=warmup_iters)
    parser.add_argument("--cosine-cycle-iters", type=int, default=cosine_cycle_iters, help="衰减终点，未设置时使用 total_steps")
    parser.add_argument("--beta1", type=float, default=betas[0])
    parser.add_argument("--beta2", type=float, default=betas[1])
    parser.add_argument("--eps", type=float, default=eps)
    parser.add_argument("--weight-decay", type=float, default=weight_decay)
    parser.add_argument("--max-l2-norm", type=float, default=max_l2_norm)
    parser.add_argument("--log-interval", type=int, default=log_interval)
    parser.add_argument("--eval-interval", type=int, default=eval_interval)
    parser.add_argument("--eval-batches", type=int, default=eval_batches)
    parser.add_argument("--save-interval", type=int, default=save_interval)
    args = parser.parse_args()
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.total_steps

    # 这些配置会用于取余、采样和除法，提前报告无效值。
    for name in (
        "vocab_size", "context_length", "d_model", "d_ff", "num_layers",
        "num_heads", "batch_size", "total_steps", "log_interval",
        "eval_interval", "eval_batches", "save_interval",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    if args.d_model % args.num_heads or (args.d_model // args.num_heads) % 2:
        parser.error("d_model 必须能被 num_heads 整除，且每个头的维度必须为偶数")
    if not 0 <= args.warmup_iters < args.cosine_cycle_iters:
        parser.error("需要 0 <= warmup_iters < cosine_cycle_iters")
    if not 0 <= args.min_learning_rate <= args.max_learning_rate:
        parser.error("需要 0 <= min_learning_rate <= max_learning_rate")
    if not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1):
        parser.error("beta1 和 beta2 必须在 [0, 1) 内")
    if args.eps <= 0 or args.max_l2_norm <= 0 or args.rope_theta <= 0 or args.weight_decay < 0:
        parser.error("eps、max_l2_norm、rope_theta 必须大于 0，weight_decay 不能为负")
    return args


def main() -> None:
    args = parse_args()

    # ---------- 5. 随机种子、数据和保存目录 ----------
    # NumPy 控制 batch 采样；PyTorch 控制模型参数初始化。
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = np.memmap(args.train_data, dtype=args.data_dtype, mode="r")
    valid_data = np.memmap(args.valid_data, dtype=args.data_dtype, mode="r")
    for name, data in (("训练集", train_data), ("验证集", valid_data)):
        if len(data) <= args.context_length:
            raise ValueError(f"{name}至少需要 context_length + 1 个 token，当前只有 {len(data)} 个")

    # 默认使用顶部配置的路径；--output-dir 可将调试存档保存到另一目录。
    latest_path = checkpoint_path if args.output_dir is None else args.output_dir / "latest.pt"
    final_path = final_checkpoint_path if args.output_dir is None else args.output_dir / "final.pt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 6. 创建模型和优化器，再恢复已有状态 ----------
    transformer = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=dtype,
    )
    optimizer = AdamW(
        transformer.parameters(),
        lr=args.max_learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    start_step = 0
    if args.resume is not None:
        # 恢复权重，以及 AdamW 每个参数的 m、v、t；不能只恢复模型。
        start_step = load_checkpoint(args.resume, transformer, optimizer)
        if not 0 <= start_step <= args.total_steps:
            raise ValueError(f"存档已完成 {start_step} 步，total_steps 不能小于此值")
        print(f"已恢复 {args.resume}，已完成 {start_step} 步")

    # ---------- 7. 训练、日志、验证和定期保存 ----------
    transformer.train()
    completed_steps = start_step
    for step in range(start_step, args.total_steps):
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        inputs, targets = get_batch(
            train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )  # inputs、targets：(B, T)，元素都是 token ID。
        optimizer.zero_grad()
        logits = transformer(inputs)  # (B, T, V)：各位置对词表中所有 token 的预测。
        loss = cross_entropy(logits=logits, targets=targets)  # 标量：B*T 个位置的平均 loss。
        loss.backward()
        gradient_clipping(transformer.parameters(), max_l2_norm=args.max_l2_norm)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        completed_steps = step + 1  # 存档记录已完成次数；恢复时从这个索引继续。

        if completed_steps % args.log_interval == 0:
            print(f"完成步数: {completed_steps}，训练 loss: {loss.item():.4f}，学习率: {lr:.6g}")

        if completed_steps % args.eval_interval == 0:
            transformer.eval()
            with torch.no_grad():
                total_eval_loss = 0.0
                for _ in range(args.eval_batches):
                    eval_inputs, eval_targets = get_batch(
                        valid_data,
                        batch_size=args.batch_size,
                        context_length=args.context_length,
                        device=args.device,
                    )
                    eval_logits = transformer(eval_inputs)
                    loss_eval = cross_entropy(eval_logits, targets=eval_targets)
                    total_eval_loss += loss_eval.item()
                avg_eval_loss = total_eval_loss / args.eval_batches
                print(f"完成步数: {completed_steps}，平均验证 loss: {avg_eval_loss:.4f}")
            transformer.train()

        if completed_steps % args.save_interval == 0:
            save_checkpoint(transformer, optimizer, iteration=completed_steps, out=latest_path)
            print(f"已保存 {latest_path}，已完成 {completed_steps} 步")

    # ---------- 8. 结束时保存：即使最后一步没到保存间隔，也保留最终状态 ----------
    save_checkpoint(transformer, optimizer, iteration=completed_steps, out=final_path)
    print(f"训练结束，已完成 {completed_steps} 步，最终存档: {final_path}")


if __name__ == "__main__":
    main()
