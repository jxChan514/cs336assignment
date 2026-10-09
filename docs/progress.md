# 学习进度

最后更新：2026-10-09。记录当前公开源码中的实现与已执行验证。

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
- [x] 生成模块框架：`sample_next_token` 接口定义、张量形状说明与固定随机种子的调试示例，语法检查通过。
- [x] 温度 / top-p 采样：评分缩放、排序与累积概率筛选、重新归一化、随机采样与 token ID 映射，4 项独立检查通过。

生成模块本次还收录了 `generate` 循环源码和小模型调试入口，包含上下文截取、逐 token 采样与序列拼接。

- [x] 单条生成序列返回与 checkpoint 文本生成，小样本 CPU 流程通过。
- [x] 保存 Tokenizer 词表与合并规则；训练接入配置、CSV、累计耗时、笔记与曲线。
- [x] 并行数据准备：EOS 分块、BPE 统计、缓存编码与 uint16 输出，小样本对照通过。
- [x] 学习率 / batch size 实验脚本与汇总工具，CPU benchmark 及已记录结果的汇总检查通过。
- [x] matplotlib 依赖与锁定文件更新，离线锁定检查通过。

## 验证记录

2026-10-09：在 macOS / Python 3.12 下运行完整 `pytest -q --tb=short`，得到 **51 passed，2 skipped**。两项跳过为官方 macOS 内存限制测试。同步的 10 个 Python 文件通过语法检查，`uv lock --check --offline` 通过。

本次额外执行以下本地检查：

- 单条序列生成：固定 logits 检查生成 ID、长度、上下文截取、零步返回和遇 EOS 停止；加载小样本 checkpoint 后编码、生成和解码文本通过。此处验证范围为单条输入。
- 数据准备与训练：保存词表 / 合并规则，CPU 训练到第 4 步并恢复到第 6 步；CSV 步数为 1–6，未验证步的 valid_loss 为空，累计耗时递增；配置与步数 / 时间两张 PNG 生成成功。
- 并行正式数据准备：2 个 worker、小样本、300 词词表，uint16 编码解码完整往返；生成的词表与合并规则与原 `train_bpe` 结果一致。
- 学习率和 batch size 脚本：CPU、float32、小模型各运行 4 步 benchmark，正常结束并保存 summary。这些检查覆盖本地 benchmark 分支。
- 结果汇总：读取已有五组学习率实验记录，生成 comparison JSON 和两张对照曲线，按最终验证 loss 选出的学习率为 0.003。已有 GPU 记录及其来源见 [实验记录](experiments.md)，本次同步未重新运行完整 GPU 实验。


2026-10-08（第三次同步）：对 `sample_next_token` 执行 4 项独立检查，批量输出形状 / dtype / token 范围、top-p 候选筛选及采样频率、小 top-p 下保留最高概率 token、温度缩放后的采样频率均通过。频率检查固定随机种子为 42，每组采样 5000 次，以 0.03 的绝对误差与预期分布比较。本次采样验证针对 `sample_next_token`。已有仓库测试回归为 **51 passed，2 skipped**，覆盖 BPE、Tokenizer、模型组件、训练工具、采样数据与存档。

2026-10-08（第二次同步）：新增 `generation.py` 的接口框架与调试示例，`python -m py_compile cs336_basics/generation.py` 语法检查通过。本次验证范围为文件语法。

2026-10-08（第一次同步）：在 macOS / Python 3.12 下运行完整 `pytest -q` 测试集，得到 **51 passed，2 skipped**。新增数据采样与 checkpoint 官方测试均通过；两项跳过是官方在 macOS 上禁用的内存限制测试。

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
