# A5 RL 变体正式源码

来源：2026-10-29 固定 3-shot 实验的服务器归档。`protocol.json` 包含十个方法的完整数值配置；`code/train.py --method` 选择单一方法。

- `code/objectives.py`：组基线、优势归一化、软化 MaxRL 和 on-policy loss。
- `code/losses.py`：naive、逐 token noclip、PPO clip、序列级 GSPO。
- `code/train.py`：采样、冻结 old log probs、梯度累积、验证和 W&B。
- `code/scoring.py`：仅回答位置的分块 log probability / entropy。
- `code/reference/grpo.py`：token IDs、右侧 padding、shifted labels 与 response mask。
- `code/server.py`：双卡 vLLM 生成与 NCCL 权重同步。
- `code/baseline.py`、`code/aggregate.py`：原模型评测与已有运行汇总。

这组实验学习率为 `1e-5`，不是后来 DAPO / OPD 的 `1e-6`。on-policy 每次采 32 题 × 8 回答并更新一次；off-policy 每批 256 回答分成 32 次 8 回答更新，冻结整个 rollout 的旧概率。每种方法独立从原始预训练权重开始。

部分 off-policy / 自定义方法退化后提前停止，不应被描述为全部完成 200 轮。配置记录目标预算，真实完成量应查看服务器归档的终态日志。CPU 数值测试可以从[上级目录](../README.md)运行，GPU 启动前仍须准备数据、模型、环境及路径；这里不包含数据与运行回执。
