# CS336 Assignments · 持续更新中

Stanford CS336 **Language Modeling from Scratch** 的个人学习实现与实验记录，基于 Spring 2026 作业版本，随学习进度持续更新。

**最近更新：2026-10-09** · 本次更新：单条文本生成、实验日志与 loss 曲线、并行数据准备、学习率 / batch size 实验脚本，以及 matplotlib 依赖。

这是个人自学仓库，记录已更新的实现、实验与验证结果。

## 已更新作业

| 作业 | 状态 | 当前内容 |
| --- | --- | --- |
| [A1 · Basics](assignments/assignment1-basics/) | 已更新 | BPE、Tokenizer、TransformerLM、交叉熵、AdamW、学习率调度、梯度裁剪、数据准备与采样、训练与验证、checkpoint、SGD 学习率实验 |

## Assignment 1

| 模块 | 状态 | 入口 |
| --- | --- | --- |
| BPE 训练 | 已实现，相关测试通过 | [bpe.py](assignments/assignment1-basics/cs336_basics/bpe.py) |
| Tokenizer | 已实现，功能测试通过 | [tokenizer.py](assignments/assignment1-basics/cs336_basics/tokenizer.py) |
| Linear / Embedding / RMSNorm | 已实现，相关测试通过 | [model.py](assignments/assignment1-basics/cs336_basics/model.py) |
| SiLU / SwiGLU / RoPE | 已实现，相关测试通过 | [model.py](assignments/assignment1-basics/cs336_basics/model.py) |
| Softmax / Scaled dot-product attention | 已实现，相关测试通过 | [model.py](assignments/assignment1-basics/cs336_basics/model.py) |
| Multi-head self-attention（含可选 RoPE） | 已实现，含 / 不含 RoPE 的测试均通过 | [model.py](assignments/assignment1-basics/cs336_basics/model.py) |
| TransformerBlock | 已实现，相关测试通过 | [model.py](assignments/assignment1-basics/cs336_basics/model.py) |
| TransformerLM 前向计算 | 已实现，正常输入与截短输入测试均通过 | [model.py](assignments/assignment1-basics/cs336_basics/model.py) |
| Cross-entropy | 已实现，官方测试通过，包含大 logits 数值稳定性检查 | [model.py](assignments/assignment1-basics/cs336_basics/model.py) |
| AdamW | 已实现，官方测试通过 | [optimizer.py](assignments/assignment1-basics/cs336_basics/optimizer.py) |
| Warmup + cosine learning-rate schedule | 已实现，官方测试通过 | [optimizer.py](assignments/assignment1-basics/cs336_basics/optimizer.py) |
| Gradient clipping | 已实现，官方测试通过 | [optimizer.py](assignments/assignment1-basics/cs336_basics/optimizer.py) |
| SGD 学习率实验 | 已运行，比较 10 / 100 / 1000 三组学习率 | [learning_rate_experiment.py](assignments/assignment1-basics/cs336_basics/learning_rate_experiment.py) |
| 数据采样 | 已实现，官方测试通过 | [data.py](assignments/assignment1-basics/cs336_basics/data.py) |
| Checkpoint 保存与加载 | 已实现，官方测试通过 | [checkpoint.py](assignments/assignment1-basics/cs336_basics/checkpoint.py) |
| TinyStories 样本准备 | 已运行，按完整故事划分、训练 BPE 并编码为二进制 token 数据 | [prepare_data.py](assignments/assignment1-basics/cs336_basics/prepare_data.py) |
| 训练 / 验证 / 断点恢复 | 已实现，小样本 CPU 训练与恢复流程验证通过 | [train.py](assignments/assignment1-basics/cs336_basics/train.py) |
| 温度 / top-p 采样 | 已实现，形状、候选筛选、频率与温度检查通过 | [generation.py](assignments/assignment1-basics/cs336_basics/generation.py) |
| 单条序列生成 | 返回完整 token 序列，单条输入的上下文截取、零步生成与 EOS 停止检查通过 | [generation.py](assignments/assignment1-basics/cs336_basics/generation.py) |
| Checkpoint 文本生成 | 加载 Tokenizer 和模型存档，编码 prompt 并解码生成结果，小样本 CPU 流程通过 | [generate_text.py](assignments/assignment1-basics/cs336_basics/generate_text.py) |
| 实验日志 / loss 曲线 | 记录配置、CSV、累计耗时与实验笔记，训练恢复及两种横轴绘图通过 | [experiment.py](assignments/assignment1-basics/cs336_basics/experiment.py) / [plot_metrics.py](assignments/assignment1-basics/cs336_basics/plot_metrics.py) |
| 并行实验数据准备 | 按 EOS 边界分块、统计 BPE、缓存编码并输出 uint16 数据；小样本往返和 BPE 对照通过 | [prepare_experiment_data.py](assignments/assignment1-basics/cs336_basics/prepare_experiment_data.py) |
| 学习率 / batch size 实验 | W&B、失稳检测、token 预算、随机状态存档与恢复；两个脚本的 CPU benchmark 短流程通过 | [run_lr_experiment.py](assignments/assignment1-basics/cs336_basics/run_lr_experiment.py) / [run_batch_experiment.py](assignments/assignment1-basics/cs336_basics/run_batch_experiment.py) |
| 学习率结果汇总 | 汇总配置与指标，生成步数 / 耗时对照曲线，已记录五组实验的汇总流程通过 | [summarize_lr_experiment.py](assignments/assignment1-basics/cs336_basics/summarize_lr_experiment.py) |

Tokenizer 支持特殊 token、UTF-8 编解码、按输入段惰性编码，以及词表和合并规则的 JSON 保存/加载。文件格式与流式分段约定见源码 docstring。

## 运行与测试

需要 Python 3.12 或 3.13，以及 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/jxChan514/cs336assignment.git
cd cs336assignment/assignments/assignment1-basics
uv sync --locked

# 运行仓库测试
uv run pytest -q
```

2026-10-08（第一次同步），在 macOS、Python 3.12 环境运行完整测试集：**51 passed，2 skipped**。其中数据采样与 checkpoint 官方测试通过。两项跳过是官方测试在 macOS 上禁用的内存限制测试。

2026-10-08（第二次同步），新增 `generation.py` 的接口框架与调试示例，使用 `python -m py_compile cs336_basics/generation.py` 完成语法检查。

2026-10-08（第三次同步），温度与 top-p 采样的 4 项独立检查通过：批量输出形状与 token 范围、候选筛选及重新归一化后的采样频率、小 top-p 下的最高概率 token 保留、温度缩放后的采样频率。这些检查针对 `sample_next_token`。已有仓库测试回归结果为 **51 passed，2 skipped**，覆盖 BPE、Tokenizer、模型组件、训练工具、采样数据与存档。

2026-10-09：完整测试集 **51 passed，2 skipped**；10 个同步 Python 文件语法检查、依赖锁定检查、短训练及恢复、CSV 与曲线、单条 checkpoint 文本生成、并行数据准备、两个实验脚本的 CPU benchmark 与学习率汇总检查通过。完整验证范围见 [进度记录](docs/progress.md#验证记录)。

## 数据准备与训练

在作业目录中运行以下命令，使用仓库自带 TinyStories 样本完成一次短训练及恢复验证：

```bash
uv run python cs336_basics/prepare_data.py

uv run python cs336_basics/train.py \
  --total-steps 4 --warmup-iters 1 --cosine-cycle-iters 6 \
  --log-interval 1 --eval-interval 2 --eval-batches 2 --save-interval 2 \
  --output-dir checkpoints/smoke-initial

uv run python cs336_basics/train.py \
  --resume checkpoints/smoke-initial/final.pt \
  --total-steps 6 --warmup-iters 1 --cosine-cycle-iters 6 \
  --log-interval 1 --eval-interval 2 --eval-batches 2 --save-interval 2 \
  --output-dir checkpoints/smoke-resumed
```

样本按完整故事划分为 4 篇训练、1 篇验证，并生成 `data/train.bin` 与 `data/valid.bin`。训练脚本支持数据路径、模型规模、设备、学习率与日志间隔等命令行配置；checkpoint 保存模型、优化器状态和已完成步数。上述流程已验证从第 4 步恢复并训练到第 6 步。

## 学习率实验

在同一目录运行 SGD 学习率实验：

```bash
uv run python cs336_basics/learning_rate_experiment.py
```

该实验固定初始权重，在二次损失上比较三组学习率；本次运行中，10 与 100 的损失下降，1000 的损失快速增大。具体结果见 [验证记录](docs/progress.md#验证记录)。

## 实验记录与对照

训练脚本现在自动保存配置、指标 CSV、实验笔记和 loss 曲线。`prepare_data.py` 同时保存词表与合并规则，供 `generate_text.py` 加载。学习率与 batch size 实验的入口、五组已记录的学习率结果见 [实验记录](docs/experiments.md)。

## 仓库结构

```text
assignments/assignment1-basics/
  cs336_basics/       # 个人实现与官方预分词示例
  tests/             # 官方测试、测试夹具与补充测试
  pyproject.toml     # 依赖声明（含 matplotlib）
  uv.lock            # 对应依赖锁定文件
  LICENSE            # 原始 Stanford MIT 许可证
docs/
  progress.md        # 已更新模块与验证记录
  upstream.md        # 官方来源、版本与修改范围
CHANGELOG.md         # 仓库更新记录
```

## 持续更新

随学习进度同步代码，并更新 [进度记录](docs/progress.md) 与 [更新日志](CHANGELOG.md)。发布说明记录已经更新的模块、实验和验证结果。

## 来源与许可

课程主页：[Stanford CS336](https://cs336.stanford.edu/)。起始代码、测试及作业说明来自 [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)，具体版本与修改范围见 [upstream.md](docs/upstream.md)。

本仓库采用 [MIT License](LICENSE)，上游文件保留原有版权与许可证。
