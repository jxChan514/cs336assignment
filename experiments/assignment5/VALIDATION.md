# 发布前验证 · 2026-11-03

从 Minimind 已归档的代码和配置提取源码，未修改归档文件、运行真实模型、重新生成回答或重跑训练。

| 检查 | 结果 |
| --- | --- |
| RL 变体优势、loss、解析/数值梯度、零优势剪枝与累积 | 5 个 unittest 通过 |
| 回答位置分块打分对照完整输出头 | 通过；最大参数梯度差 `3.5762786865234375e-07` |
| DAPO 筛组、奖励、非对称裁剪、完整批次 token 分母、warmup、恢复谱系及检查点恢复/轮换 | 8 个 unittest 通过 |
| OPD 完整词表 KL 独立 oracle / gradcheck、教师冻结、分块与不等长 microbatch、EOS/mask、Adam/RNG/下一更新恢复 | 4 个 unittest 通过 |
| Python 语法 | 45 个文件通过 |
| JSON、shell 语法和 README 本地链接 | 通过 |
| 归档文件 SHA-256 与源码清单 | 60 个提取文件核对；其中 2 个配置的导出脱敏前后哈希均有记录 |
| 候选发布文件中的凭据及生成产物检查 | 通过；不包含原始回答、日志、数据集或模型参数 |

CPU 数值测试使用 Minimind 已有的 Python 3.12、PyTorch `2.10.0+cu129`、Transformers `4.57.6` 环境，并设置 `CUDA_VISIBLE_DEVICES=""`、`HF_HUB_OFFLINE=1`。测试中的小型随机模型与临时检查点用于验证公式和恢复行为，不是正式训练。复用入口为 `test_cpu.sh`。

正式 GPU 训练的历史环境版本见各 `env-spec.json` 和总说明。这里的发布前检查没有重新验证 CUDA / vLLM 0.22 / NCCL 的硬件兼容性，也不是对课程完整测试集的再次运行。评分器保留上游正则字符串的 Python SyntaxWarning；源码语法检查通过，没有为发布改写评分逻辑。
