# 学习进度

最后更新：2026-10-08。记录当前公开源码中的实现与已执行验证。

## Assignment 1 · Basics

- [x] 字节级 BPE 训练：预分词、pair 统计、合并与特殊 token 边界。
- [x] Tokenizer：encode / decode、特殊 token、按段 encode_iterable。
- [x] Tokenizer 文件接口：JSON 保存与加载，保留任意字节。
- [x] Tokenizer 边界测试：重叠特殊 token、非法 UTF-8、稀疏 ID、文件往返、惰性迭代。
- [x] Linear、Embedding、RMSNorm、SiLU、SwiGLU、RoPE。
- [x] 数值稳定的 Softmax，相关测试通过。
- [x] Scaled dot-product attention，标准输入与四维输入测试通过。
- [x] Multi-head self-attention：Q/K/V 投影、拆头、因果 mask、合并各头与输出投影，官方测试通过。
- [x] 在 Multi-head self-attention 中接入可选 RoPE，官方测试通过。
- [x] TransformerBlock：RMSNorm、带 RoPE 的注意力、SwiGLU 与残差连接，官方测试通过。
- [x] 完整 TransformerLM 前向计算：token embedding、多层 TransformerBlock、最终 RMSNorm 与输出投影，正常输入与截短输入的官方测试通过。
- [x] Cross-entropy：平均交叉熵损失，官方测试通过，包含大 logits 的数值稳定性检查。
- [x] AdamW：一阶 / 二阶矩估计、偏差修正与解耦权重衰减，官方测试通过。
- [x] 学习率调度：线性 warmup、余弦衰减与末尾最小学习率，官方测试通过。
- [x] 全局 L2 梯度裁剪：跳过没有梯度的参数，官方测试通过。
- [x] SGD 学习率实验：固定初始权重，比较 10 / 100 / 1000 三组学习率，脚本运行完成。
- [x] 数据采样：连续 token 输入、向后偏移一位的目标与设备放置，官方测试通过。
- [x] Checkpoint：模型、优化器状态与已完成步数的保存和加载，官方测试通过。
- [x] TinyStories 样本准备：按完整故事划分数据、训练 BPE、编码并保存 uint16 token 文件。
- [x] 训练流程：前向与反向计算、梯度裁剪、学习率调度、AdamW 更新、日志、验证与存档。
- [x] 命令行训练配置与断点恢复：CPU 小样本训练到第 4 步，恢复后继续到第 6 步。

## 验证记录

2026-10-08：在 macOS / Python 3.12 下运行完整 `pytest -q` 测试集，得到 **51 passed，2 skipped**。新增数据采样与 checkpoint 官方测试均通过；两项跳过是官方在 macOS 上禁用的内存限制测试。

同日运行 README 中的数据准备与短训练流程：样本包含 5 篇故事，前 4 篇用于训练，最后 1 篇用于验证。训练使用 CPU、float32、batch size 2、context length 32、d_model 64、2 层与 4 个注意力头；warmup 为 1 步，余弦衰减终点为第 6 步。

首次训练到第 4 步，训练 loss 为 6.7193、平均验证 loss 为 6.7055，并保存模型、优化器和步数。随后从存档恢复，完成第 5、6 步更新；第 6 步训练 loss 为 6.6581、平均验证 loss 为 6.7319，成功保存最新与最终存档。这次验证覆盖了数据准备、训练、验证、存档和恢复流程。

2026-10-07：README 中列出的已实现模块测试在 macOS / Python 3.12 下得到 **49 passed，2 skipped**。新增 AdamW、余弦学习率调度与梯度裁剪的官方测试全部通过；此前已实现模块的测试全部通过。两项跳过仍为官方在 macOS 上禁用的内存限制测试。

同日运行 `uv run python cs336_basics/learning_rate_experiment.py` 对应的实验脚本，三组学习率的初始损失均约为 24.1693。第 10 次打印的损失分别为：学习率 10 → 3.24823，100 → 2.84899e-23，1000 → 2.24017e+18。这些损失在每次更新前记录，反映该二次损失实验的行为。

2026-10-04（第二次同步）：README 中列出的已实现模块测试在 macOS / Python 3.12 下得到 **46 passed，2 skipped**。新增交叉熵的官方测试通过，包含普通 logits 和大 logits 与 PyTorch 结果的一致性检查；先前已实现模块的测试全部通过。两项跳过仍为官方在 macOS 上禁用的内存限制测试。

2026-10-04（第一次同步）：README 中列出的已实现模块测试在 macOS / Python 3.12 下得到 **45 passed，2 skipped**。本次新增的 TransformerLM 正常输入与截短输入测试均通过，`test_model.py` 的全部 13 项模型测试通过；先前已实现模块的测试全部通过。两项跳过仍为官方在 macOS 上禁用的内存限制测试。

2026-10-03：README 中列出的已实现模块测试在 macOS / Python 3.12 下得到 **43 passed，2 skipped**。本次新增的带 RoPE 多头注意力与 TransformerBlock 官方测试均通过；先前已实现模块的测试全部通过。两项跳过仍为官方在 macOS 上禁用的内存限制测试。

2026-10-02：README 中列出的已实现模块测试在 macOS / Python 3.12 下得到 **41 passed，2 skipped**。其中包含 3 项 BPE 测试、23 项官方 Tokenizer 功能测试、5 项补充测试、9 项模型组件测试和 1 项 Softmax 测试；另外 2 项官方内存测试由平台条件跳过。本次新增的多头因果自注意力官方测试通过。

2026-10-01（第二次同步）：README 中列出的已实现模块测试在 macOS / Python 3.12 下得到 **40 passed，2 skipped**。其中包含 3 项 BPE 测试、23 项官方 Tokenizer 功能测试、5 项补充测试、8 项模型组件测试和 1 项 Softmax 测试；另外 2 项官方内存测试由平台条件跳过。本次新增的 Softmax 与两项缩放点积注意力测试均通过。

2026-10-01（首次发布）：当时已有模块的测试结果为 **37 passed，2 skipped**。

## 更新记录模板

每次更新记录：日期、本次更新的内容、验证命令与结果。
