# 数学推理强化学习与在线蒸馏

基于 OLMo-2-0425-1B 和 GSM8K，实现 GRPO、Dr.GRPO、RFT、MaxRL、软化 MaxRL、off-policy naive/noclip/clip/GSPO，以及 DAPO 和 GKD 风格 OPD。

## 代码

| 文件 | 用途 |
| --- | --- |
| `train.py` / `evaluate.py` | 统一训练入口与固定 3-shot 评测 |
| `algorithms.py` | 组优势、策略梯度、重要性权重、裁剪与 DAPO 动态筛组 |
| `distillation.py` | 学生前缀上的完整词表 reverse KL，教师冻结 |
| `model_utils.py` | 采样 token IDs、标签与回答掩码、分块 log probability |
| `sampler.py` | vLLM 生成与策略权重同步 |
| `reward.py` / `checkpoints.py` | 数学答案评分与训练状态保存/恢复 |
| `configs/` / `prompt.prompt` | 实验超参数与固定提示模板 |

## 运行

使用 Python 3.12、Linux/CUDA 双 GPU。安装 `requirements.txt`；数据使用 GSM8K 的 `train.jsonl`、`test.jsonl`，每行含 `question` 和带 `####` 最终答案的 `answer`。模型参数可填写本地目录或 Hugging Face 模型 ID。

```bash
cd experiments/assignment5
pip install -r requirements.txt

python train.py --method grpo --train-file data/train.jsonl --val-file data/test.jsonl --output outputs/grpo
python train.py --method dapo --train-file data/train.jsonl --val-file data/test.jsonl --output outputs/dapo
python train.py --method opd  --train-file data/train.jsonl --val-file data/test.jsonl --output outputs/opd --save-model

python evaluate.py --model outputs/opd/final-model --val-file data/test.jsonl --output outputs/eval
bash test_cpu.sh
```

其他 RL 方法替换 `--method`；模型位置用 `--model` / `--teacher` 指定。加 `--save-model` 导出 best/final 推理模型，续训使用 `--resume outputs/<method>/checkpoint.pt`。需要 W&B 时添加 `--wandb-project <project>`，使用自己的登录配置。

## 设置与结果

固定 3-shot、训练 6,400 题、验证 1,024 题，seed0；最多生成 512 token。

- RL 变体：每轮 32 题 × 8 回答；on-policy 每轮更新一次，off-policy 分成 32 次 8 回答更新，学习率 `1e-5`。
- DAPO：动态筛选 32 个混合 G=8 组，每轮四次 64 回答更新；裁剪 `[0.8,1.28]`，token 归一化与长度奖励；200 轮 / 800 次更新。
- OPD：冻结 OLMo-2-1124-7B-Instruct，每次 64 题各生成一条新回答并更新一次；完整词表 `KL(student || teacher)`，不按正确性筛选；800 次更新。
- DAPO / OPD 学习率 `1e-6`，80 次更新 warmup；microbatch 分别为 8 / 4。

| 方法 | 最终准确率 | 历史峰值 | 状态 |
| --- | ---: | ---: | --- |
| 原始 1B | 20.51% | — | 基线 |
| GRPO | 23.44% | 36.33% | 完成 |
| GRPO_constant | 26.95% | 36.04% | 完成 |
| Dr.GRPO | 24.51% | 40.43% | 完成 |
| RFT | 0.00% | 30.57% | 完成，后期生成退化 |
| MaxRL | 23.93% | 34.08% | 完成 |
| 软化 MaxRL（α=0.5） | 32.13% | 36.52% | 完成 |
| off-policy naive / noclip / clip / GSPO | — | — | 中断或提前停止，最后验证均为 0% |
| DAPO | **53.91%** | 54.49% | 完成 |
| OPD | **57.52%** | 57.52% | 完成 |
| 7B 教师 | 79.39% | — | 独立评测 |

DAPO / OPD 相比基线分别提升 33.40 / 37.01 个百分点。历史方法出现重复生成、截断和准确率回落；DAPO / OPD 的最终截断率为 0.39% / 0.78%。各方法预算、学习率及教师监督不同，这些单种子结果不能作为等算力算法消融。

上述数值来自已完成实验；本目录合并公共函数、去除部署依赖，整理后仅做 CPU 数值与接口验证，没有重跑 GPU 实验。生成的数据、指标和权重均不纳入源码。

参考：[CS336 A5](https://github.com/stanford-cs336/assignment5-alignment)、[DAPO](https://arxiv.org/abs/2503.14476)、[GKD](https://arxiv.org/abs/2306.13649)。数学评分器保留上游 Apache-2.0 版权说明。
