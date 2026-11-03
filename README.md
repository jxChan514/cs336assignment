# CS336 Assignments

Stanford CS336 的个人实现与实验，包含语言模型预训练、数学推理强化学习和在线蒸馏。

| 项目 | 内容 | 入口 |
| --- | --- | --- |
| Assignment 1 | BPE、Transformer、AdamW 与预训练 | [代码与运行说明](assignments/assignment1-basics/) |
| Assignment 5 | GRPO / Dr.GRPO / RFT / MaxRL、off-policy 变体、DAPO 与 GKD 风格 OPD | [实验代码、配置与结果](experiments/assignment5/) |

A5 使用 OLMo-2-0425-1B 和 GSM8K，固定 3-shot、1,024 题验证集、seed0。原始模型准确率 **20.51%**，DAPO / OPD 最终为 **53.91% / 57.52%**。OPD 使用冻结的 7B 教师；不同方法的训练预算和监督不同，完整结果及运行方式见 A5 说明。

课程基础实现、适配器和官方测试保留在 `assignments/`。公开实验目录仅包含运行所需源码、配置、提示模板和测试，不包含数据集、原始回答、运行日志、模型权重或私人部署配置。

课程来源：[Stanford CS336](https://cs336.stanford.edu/)、[Assignment 1](https://github.com/stanford-cs336/assignment1-basics)、[Assignment 5](https://github.com/stanford-cs336/assignment5-alignment)。本仓库采用 [MIT License](LICENSE)，上游文件保留各自许可证和版权说明。
