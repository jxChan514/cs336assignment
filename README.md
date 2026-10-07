# CS336 Assignments · 持续更新中

Stanford CS336 **Language Modeling from Scratch** 的个人学习实现与实验记录，基于 Spring 2026 作业版本，随学习进度持续更新。

**最近更新：2026-10-07** · 当前进度：Assignment 1 的 BPE、Tokenizer、完整 TransformerLM 前向计算、交叉熵损失、AdamW、带预热的余弦学习率调度与梯度裁剪，并加入 SGD 学习率实验。

这是个人自学仓库，包含尚未完成的作业接口。各模块的实现和验证情况见下表。

## 作业进度

| 作业 | 状态 | 当前内容 |
| --- | --- | --- |
| [A1 · Basics](assignments/assignment1-basics/) | 进行中 | BPE、Tokenizer、TransformerLM 及基础组件、Cross-entropy、AdamW、余弦学习率调度、梯度裁剪、SGD 学习率实验 |
| A2 · Systems | 待开始 | 后续补充 GPU kernel、性能分析与分布式训练 |
| A3 · Scaling | 待开始 | 后续补充 scaling laws 实验 |
| A4 · Data | 待开始 | 后续补充数据处理与过滤 |
| A5 · Alignment | 待开始 | 后续补充对齐与强化学习 |

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
| 数据采样 / 训练 / Checkpoint | 待实现 | [adapters.py](assignments/assignment1-basics/tests/adapters.py) |

Tokenizer 支持特殊 token、UTF-8 编解码、按输入段惰性编码，以及词表和合并规则的 JSON 保存/加载。文件格式与流式分段约定见源码 docstring。

## 运行与测试

需要 Python 3.12 或 3.13，以及 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/jxChan514/cs336assignment.git
cd cs336assignment/assignments/assignment1-basics
uv sync --locked

# 已实现模块的测试
uv run pytest -q \
  tests/test_train_bpe.py \
  tests/test_tokenizer.py \
  tests/test_tokenizer_extra.py \
  tests/test_model.py \
  tests/test_nn_utils.py \
  tests/test_optimizer.py
```

2026-10-07，在 macOS、Python 3.12 环境验证上述测试：**49 passed，2 skipped**。本次新增的 AdamW、余弦学习率调度与梯度裁剪官方测试全部通过。两项跳过是官方测试在 macOS 上禁用的内存限制测试，尚未验证 Linux 下的内存限制行为。

在同一目录运行 SGD 学习率实验：

```bash
uv run python cs336_basics/learning_rate_experiment.py
```

该实验固定初始权重，在二次损失上比较三组学习率；本次运行中，10 与 100 的损失下降，1000 的损失快速增大。具体结果见 [验证记录](docs/progress.md#验证记录)。

运行 `uv run pytest` 可检查完整作业；未完成的接口仍会触发 `NotImplementedError`，当前不宣称完整作业测试通过。大型语料训练和端到端语言模型实验尚未完成。

## 仓库结构

```text
assignments/assignment1-basics/
  cs336_basics/       # 个人实现与官方预分词示例
  tests/             # 官方测试、测试夹具与补充测试
  pyproject.toml     # 官方依赖声明
  uv.lock            # 官方依赖锁定文件
  LICENSE            # 原始 Stanford MIT 许可证
docs/
  progress.md        # 学习进度与后续计划
  upstream.md        # 官方来源、版本与修改范围
CHANGELOG.md         # 仓库更新记录
```

## 持续更新

后续按学习进度补充实现、测试和实验结果，同时更新 [进度记录](docs/progress.md) 与 [更新日志](CHANGELOG.md)。每次提交说明本次完成的模块和验证情况。

## 来源与许可

课程主页：[Stanford CS336](https://cs336.stanford.edu/)。起始代码、测试及作业说明来自 [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)，具体版本与修改范围见 [upstream.md](docs/upstream.md)。

本仓库采用 [MIT License](LICENSE)，上游文件保留原有版权与许可证。
