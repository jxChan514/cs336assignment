# 更新日志

本仓库持续更新，记录每次更新的实现、实验与验证结果。

## 2026-10-08

- 新增 `data.py`，随机采样连续 token 序列及向后偏移一位的预测目标，支持指定设备。
- 新增 `checkpoint.py`，保存和加载模型、优化器状态及已完成步数，支持路径与二进制文件对象。
- 新增 `prepare_data.py`，按完整故事划分 TinyStories 样本，在训练集上学习 BPE 并生成训练 / 验证 token 二进制文件。
- 新增 `train.py`，串联采样、前向计算、交叉熵、反向传播、梯度裁剪、学习率调度、参数更新、验证与定期存档，支持命令行配置和断点恢复。
- 接入数据采样与 checkpoint 测试适配器，并同步交叉熵的张量形状与计算流程注释。
- 完整测试集：51 passed，2 skipped（macOS / Python 3.12）；小样本 CPU 训练完成 4 步，并成功恢复到第 6 步。
- 更新 README、学习进度与来源说明，发布文档集中记录已经更新的内容和验证结果。

## 2026-10-07

- 新增 `optimizer.py`：AdamW、线性预热与余弦衰减学习率调度、全局 L2 梯度裁剪。
- 接入对应官方测试适配器；3 项新增功能测试通过，已有实现回归测试合计：49 passed，2 skipped（macOS / Python 3.12）。
- 新增并运行 `learning_rate_experiment.py`，以相同初始权重比较 SGD 在学习率 10、100、1000 下的二次损失变化；1000 在本次 10 步记录中损失快速增大。
- 更新 README、学习进度与上游修改范围。

## 2026-10-04 · 第二次同步

- 新增数值稳定的 `cross_entropy`，沿词表维度计算目标 token 的负对数概率，并返回平均损失。
- 接入官方交叉熵测试适配器；与 PyTorch 结果一致，普通 logits 和大 logits 的测试均通过。
- 同步模型源码中新增的张量形状、广播、RoPE、注意力与语言模型计算流程注释。
- 已有实现回归测试合计：46 passed，2 skipped（macOS / Python 3.12）；更新 README 和学习进度。

## 2026-10-04 · 第一次同步

- 新增 `TransformerLM`，组合 token embedding、多层 TransformerBlock、最终 RMSNorm 与语言模型输出投影，返回每个位置的词表 logits。
- 在官方测试适配器中接入语言模型实现与权重加载。
- TransformerLM 正常输入与截短输入的 2 项官方测试均通过；`test_model.py` 全部 13 项测试通过，已有实现回归测试合计：45 passed，2 skipped（macOS / Python 3.12）。
- 更新 README 和学习进度。

## 2026-10-03

- 为 `MultiHeadSelfAttention` 接入可选 RoPE，支持传入 token positions，并在省略时生成默认位置。
- 将 RoPE 的频率缓存设为非持久化 buffer。
- 新增 `TransformerBlock`，组合 RMSNorm、带 RoPE 的多头注意力、SwiGLU 与残差连接。
- 接入对应官方测试适配器；新增的 2 项官方测试通过，已有实现回归测试合计：43 passed，2 skipped（macOS / Python 3.12）。
- 更新 README 和学习进度。

## 2026-10-02

- 新增 `MultiHeadSelfAttention`：Q/K/V 投影、拆头、因果 mask、合并各头与输出投影。
- 在官方测试适配器中接入无 RoPE 的多头自注意力实现。
- 新增功能的官方测试通过，已有实现回归测试合计：41 passed，2 skipped（macOS / Python 3.12）。
- 更新 README 和学习进度。

## 2026-10-01 · 第二次同步

- 新增数值稳定的 Softmax 与支持 mask 的缩放点积注意力实现。
- 在官方测试适配器中接入 Softmax 与缩放点积注意力。
- 新增功能的 3 项官方测试通过，已有实现回归测试合计：40 passed，2 skipped（macOS / Python 3.12）。
- 更新 README 和学习进度。

## 2026-10-01 · 首次发布

- 首次整理并发布 Assignment 1 的现有学习实现。
- 收录字节级 BPE 训练、Tokenizer 编解码、特殊 token 处理及 JSON 文件接口。
- 收录 Linear、Embedding、RMSNorm、SiLU、SwiGLU 和 RoPE。
- 保留官方测试、测试夹具、依赖锁定文件、作业说明和许可证，加入 5 项 Tokenizer 补充测试。
- 已实现模块在 macOS / Python 3.12 下验证：37 passed，2 skipped。
