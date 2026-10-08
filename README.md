# CS336 Assignments · 持续更新中

Stanford CS336 **Language Modeling from Scratch** 的个人学习实现与实验记录，基于 Spring 2026 作业版本，随学习进度持续更新。

**最近更新：2026-10-08** · 本次新增：生成模块框架，包含 `sample_next_token` 接口定义、输入输出形状说明与固定随机种子的调试示例。

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
| 生成模块框架 | 新增接口定义与调试示例，语法检查通过 | [generation.py](assignments/assignment1-basics/cs336_basics/generation.py) |

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

## 仓库结构

```text
assignments/assignment1-basics/
  cs336_basics/       # 个人实现与官方预分词示例
  tests/             # 官方测试、测试夹具与补充测试
  pyproject.toml     # 官方依赖声明
  uv.lock            # 官方依赖锁定文件
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
