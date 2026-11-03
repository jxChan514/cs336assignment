# Assignment 5 · RL 变体、DAPO 与 GKD 风格 OPD

这里保存实际用于 2026-10-29 至 2026-11-02 实验的算法、训练、评测、检查点和数值测试源码。源码来自服务器归档，保留正式版本；课程目录中的学习代码可以继续独立修改。DAPO 和 OPD 是作业之外的补充实验。

| 内容 | 算法入口 | 训练入口 |
| --- | --- | --- |
| GRPO、GRPO_constant、Dr.GRPO、RFT、MaxRL、软化 MaxRL | [objectives.py](rl_variants/code/objectives.py) | [train.py](rl_variants/code/train.py) |
| offpolicy_naive / noclip / clip / GSPO | [losses.py](rl_variants/code/losses.py) | 同上，通过 `--method` 选择 |
| DAPO：动态采样、非对称裁剪、token 归一化、长度奖励 | [dapo.py](dapo/code/dapo.py) | [train_dapo.py](dapo/code/train_dapo.py) |
| GKD 风格 OPD：完整词表 reverse KL | [gkd.py](gkd_opd/code/gkd.py) | [train.py](gkd_opd/code/train.py) |
| 3-shot 原始模型评测 | [baseline.py](rl_variants/code/baseline.py) | 只评测，不训练 |

每组都有 `protocol.json`、固定 3-shot `code/prompt.prompt` 和独立的 `reference/` 依赖快照。`reference/grpo.py` 提供实际采样 token IDs 的打包、标签与掩码、log probability、奖励和梯度汇总；`server.py` 实现训练卡到 vLLM 推理卡的 NCCL 权重同步与缓存刷新。`scoring.py` / OPD 的 `gkd.py` 按回答位置分块计算输出头。

## 算法差别

设同题第 $j$ 条回答的奖励为 $r_j$，组均值为 $\mu$、样本标准差为 $\sigma$。on-policy 变体用 $-A_j\log\pi_\theta(y_{jt}\mid s_{jt})$ 构造 token loss：

| 方法 | 回答权重 $A_j$ | loss 汇总 |
| --- | --- | --- |
| GRPO | $(r_j-\mu)/(\sigma+\epsilon)$ | 各回答先对自身长度取平均，再对回答取平均 |
| GRPO_constant | 同 GRPO | 回答 token 总和 / 原回答数 / 固定最大长度 |
| Dr.GRPO | $r_j-\mu$ | 同固定分母 |
| RFT | $r_j$ | 同固定分母；二值奖励下只保留正确回答的正向学习信号 |
| MaxRL | $(r_j-\mu)/(\mu+\epsilon)$ | 同固定分母 |
| 软化 MaxRL | $(r_j-\mu)/(\mu+\epsilon)^{0.5}$ | 同固定分母 |

off-policy 在一次 rollout 内冻结旧概率与组优势，再打乱回答分批更新：naive 使用旧样本上的普通对数概率目标；noclip 使用逐 token 重要性比值；clip 对其应用 PPO 式 surrogate；GSPO 使用回答内平均 log ratio 的指数作为序列权重。padding 不参与比值、长度或损失统计。

DAPO 保留原始二值奖励混合的组，补采直到得到 32 个 G=8 的有效组。之后采用 $2r-1-\operatorname{clamp}((\ell-384)/128,0,1)$ 塑形、组优势、$[0.8,1.28]$ 非对称裁剪；每轮四次 64 回答更新，每次使用完整 64 回答的有效 token 总数作分母。

OPD 每次从学生新采样 64 条回答，在这些固定前缀上最小化：

$$
L_{\mathrm{OPD}}=\frac{1}{N}\sum_{i,t}\sum_{v\in V}
p_\theta(v\mid s_{it})\left[\log p_\theta(v\mid s_{it})-\log q(v\mid s_{it})\right].
$$

教师冻结；学生概率与对数概率两条分支都参与反传。不对离散 rollout 前缀的采样过程反传。正确性奖励仅用于观察，没有进入 OPD loss。完整词表计算按 token 分块，保持全词表求和；所有 microbatch 共用整批回答 token 分母。详见 [OPD 配置与部署边界](gkd_opd/README.md)。

## 验证与运行边界

已有 Python/PyTorch/Transformers 环境中，从本目录运行以下 CPU 检查。它只运行现有的合成数值、梯度、mask、累积和检查点恢复测试，不启动 vLLM、W&B 或真实模型训练：

```bash
A5_PYTHON=/path/to/python bash test_cpu.sh
```

2026-11-03 发布前验证：17 个 unittest 及独立的分块打分/梯度对照通过；45 个 Python 文件语法通过，详见 [VALIDATION.md](VALIDATION.md)。

正式 GPU 实验使用 Linux、Python 3.12、双 A800 80GB，观测版本为 PyTorch 2.11.0、vLLM 0.22.0、Transformers 5.6.1、W&B 0.30.0。这些是独立实验快照，**不能直接套用课程目录中较早的 vLLM 依赖锁定文件**。

`code/runtime_env.sh`、配置和训练脚本保留历史服务器目录，正式启动器还检查环境/模型/教师资格/日志就绪等真实验证回执。克隆仓库后需要按自己的机器配置模型、GSM8K JSONL、数据与输出路径、环境和 W&B 登录；验证回执未纳入源码发布，不能把历史回执当作新部署的验证。此目录是可审阅的正式源码快照，不是无需资源准备的一键训练包。不要直接 source 历史环境脚本，除非部署目录与依赖确实匹配。

## 实验范围

本次正式比较均为 seed0、固定 3-shot、同一 1024 题验证集。完整运行与提前停止的方法应分别分析，源码中存在一个方法不代表它完成了预算。DAPO 完成 200 rollout 轮 / 800 参数更新；OPD 完成 800 次参数更新。它们的额外教师监督、生成量和计算开销不同，不是只改变算法的等算力消融。

这里只上传代码、配置、提示模板、测试和文件哈希，不上传原始回答、训练日志、数据集、模型、检查点或认证信息。`SOURCE_MANIFEST.json` 记录归档来源及 SHA-256；代码保持归档原样，少量配置导出时移除了连接信息或旧传输说明，并同时记录原始和发布哈希。操作策略 JSON 是历史配置记录，不代表启动现有监控或传输任务。

算法背景：[CS336 Assignment 5](https://github.com/stanford-cs336/assignment5-alignment)、[DAPO](https://arxiv.org/abs/2503.14476)、[GKD](https://arxiv.org/abs/2306.13649)。我们的 OLMo/GSM8K 配置不等同于论文全部设置。评分器及其来源说明保留在 `reference/drgrpo_grader.py` 中；仓库和上游许可证继续适用。
