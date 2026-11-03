# GKD 风格 On-Policy Distillation 正式源码

来源：2026-11-02 seed0，完成 800 次参数更新。学生为原始 OLMo-2-0425-1B，冻结教师为 OLMo-2-1124-7B-Instruct；教师 revision 固定于 `protocol.json`。两者分词器的 token-ID 映射已在原部署验证一致，使用相同原始 3-shot 文本提示，不套用各自 chat template。

- `code/gkd.py`：完整词表条件 reverse KL、教师冻结、分块输出头和整批 token 分母。
- `code/train.py`：学生新鲜回答、每批一次更新、验证、教师不变检查、HF/vLLM 同步审计。
- `code/checkpoints.py`：FP32 模型与 Adam/RNG 分盘保存，发布 size/SHA-256 清单后轮换。
- `code/history.py`：恢复后的有效更新谱系与实际总工作量。
- `code/verify_run.py`：验证真实保存的实验产物，不重新训练。
- `code/wandb_online.py`：读取实际 append-only 指标文件并在线发布；不生成指标或重启训练。

每次 64 道题、每题一条学生回答，只更新一次，再使用新学生采样下一批。microbatch 4、输出头 token chunk 128，所有 microbatch 使用整批回答 token 总数。学生 FP32 参数 / Adam + BF16 autocast；教师 BF16 / eval / no_grad。GPU 0 训练与教师评分，GPU 1 vLLM 采样。LR `1e-6`、80 更新 warmup、梯度范数 clip 1。这里的梯度裁剪与 PPO 概率比值裁剪是两回事。

监督为学生前缀上完整词表的 `KL(student || teacher)`；不对前缀采样过程求梯度，没有正确性筛选、奖励入 loss、PPO clipping 或初始参考模型 KL 惩罚。与在教师生成答案上做交叉熵 SFT 的训练分布和目标不同。

`logging-policy.json` 描述该次历史实验强制在线记录的要求。正式训练保留离线安全副本，在线上传桥最终核对全部 809 行真实历史后完成同一云端 run。上传器依赖原部署的 publication/metrics 状态文件；不能直接在新克隆中启动，也不要向该已结束 run 重传。

`train.py` 还依赖原始模型、数据、匹配的基线及教师资格验证文件；它们不包含在公开代码中。模型权重和检查点只留训练服务器，日志归档工具与认证配置不属于本次源码发布。环境版本、CPU 数值/梯度检查入口和代码哈希见[总说明](../README.md)。
