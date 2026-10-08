# 上游来源

- 课程：[Stanford CS336 · Language Modeling from Scratch](https://cs336.stanford.edu/)。
- 采用版本：Spring 2026。
- Assignment 1 上游：[stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)。
- 本次起始版本：[`a158843b20107949f1a8d7df1b05cd33b9166712`](https://github.com/stanford-cs336/assignment1-basics/commit/a158843b20107949f1a8d7df1b05cd33b9166712)。
- 上游版本号：`26.0.0`。其 README 标题仍为 Spring 2025，原文保留。
- 原始许可证：[Stanford MIT License](../assignments/assignment1-basics/LICENSE)。

## 个人实现与修改

相对上述上游版本，本次收录的实现改动为：

- 新增 `cs336_basics/bpe.py`。
- 新增 `cs336_basics/tokenizer.py`。
- 新增 `cs336_basics/model.py`。
- 新增 `cs336_basics/optimizer.py`。
- 新增 `cs336_basics/learning_rate_experiment.py`，收录 SGD 学习率比较实验。
- 新增 `cs336_basics/data.py`，提供 token 序列采样。
- 新增 `cs336_basics/checkpoint.py`，提供模型与优化器状态的保存和加载。
- 新增 `cs336_basics/prepare_data.py`，准备 TinyStories 调试样本。
- 新增 `cs336_basics/train.py`，提供训练、验证、命令行配置与断点恢复流程。
- 修改 `tests/adapters.py`，接入已经实现的模块。
- 新增 `tests/test_tokenizer_extra.py`。

官方测试、测试夹具、作业 PDF、依赖文件及其他上游文件按原样保留。`tests/fixtures/ts_tests/model.pt` 是官方测试夹具，并非本仓库训练得到的模型。其余个人训练数据、环境和运行输出不纳入公开仓库。
