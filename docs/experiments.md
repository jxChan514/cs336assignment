# Assignment 1 实验记录

2026-10-09 同步实验源码和已有结果。下表来自本地 `lr-sweep-20261009-1120` 的 config / summary 记录；本次上传前检查了汇总流程，没有重新运行完整 GPU 训练。

## 已记录的学习率实验

配置：batch size 64，context length 256，词表 10000，d_model 512，d_ff 1344，4 层、16 个头；精度 bf16，随机种子 42。完成组均训练 20,000 步、327,680,000 tokens。

| 最大学习率 | 记录状态 | 完成步数 | 最终验证 loss | 耗时（秒） |
| --- | --- | --- | --- | --- |
| 0.0003 | completed | 20,000 | 1.4661 | 3095.0 |
| 0.001 | completed | 20,000 | 1.3696 | 3110.1 |
| 0.003 | completed | 20,000 | 1.3544 | 3087.9 |
| 0.1 | completed | 20,000 | 2.0592 | 3036.7 |
| 1 | diverged | 256 | — | 38.7 |

这些记录中，学习率 0.003 的最终验证 loss 最低，为 1.3544。学习率 1 的运行记录为连续 10 步 loss 大于 100 后停止。该结论仅适用于此配置和这组记录。

## 脚本入口

在 `assignments/assignment1-basics` 中运行，路径与模型参数按实验配置指定：

```bash
# 正式数据准备（独立输出目录）
uv run python -m cs336_basics.prepare_experiment_data \
  --train-text /path/to/train.txt --valid-text /path/to/valid.txt \
  --output-dir data/tinystories --vocab-size 10000 --workers 8

# 学习率实验；在线 W&B 实体与项目由命令行指定
uv run python -m cs336_basics.run_lr_experiment \
  --data-dir data/tinystories --output-dir experiments/lr-study/gpu0-lr0.003 \
  --max-learning-rate 0.003 --wandb-mode disabled

# batch size 实验，按 token 预算安排学习率与验证
uv run python -m cs336_basics.run_batch_experiment \
  --data-dir data/tinystories --output-dir experiments/batch64 \
  --batch-size 64 --max-learning-rate 0.003 --wandb-mode disabled

# 汇总 gpu*-lr* 子目录中的已有记录
uv run python -m cs336_basics.summarize_lr_experiment experiments/lr-study
```

两个实验脚本默认使用 CUDA / bf16；本次本地短流程检查使用 CPU / float32 和 `--benchmark`。日志、W&B、完整 GPU 训练及恢复相关接口按源码收录，测试范围见 [验证记录](progress.md#验证记录)。
