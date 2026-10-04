# 学习进度

最后更新：2026-10-04。状态以当前公开源码和已执行测试为准。

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
- [ ] 数据采样、梯度裁剪。
- [ ] AdamW、学习率调度、checkpoint。
- [ ] 大型语料 BPE、模型训练、生成与实验报告。
- [ ] Linux 下官方 tokenizer 内存限制测试。

## 验证记录

2026-10-04（第二次同步）：README 中列出的已实现模块测试在 macOS / Python 3.12 下得到 **46 passed，2 skipped**。新增交叉熵的官方测试通过，包含普通 logits 和大 logits 与 PyTorch 结果的一致性检查；先前已实现模块的测试全部通过。两项跳过仍为官方在 macOS 上禁用的内存限制测试。

2026-10-04（第一次同步）：README 中列出的已实现模块测试在 macOS / Python 3.12 下得到 **45 passed，2 skipped**。本次新增的 TransformerLM 正常输入与截短输入测试均通过，`test_model.py` 的全部 13 项模型测试通过；先前已实现模块的测试全部通过。两项跳过仍为官方在 macOS 上禁用的内存限制测试。

2026-10-03：README 中列出的已实现模块测试在 macOS / Python 3.12 下得到 **43 passed，2 skipped**。本次新增的带 RoPE 多头注意力与 TransformerBlock 官方测试均通过；先前已实现模块的测试全部通过。两项跳过仍为官方在 macOS 上禁用的内存限制测试。

2026-10-02：README 中列出的已实现模块测试在 macOS / Python 3.12 下得到 **41 passed，2 skipped**。其中包含 3 项 BPE 测试、23 项官方 Tokenizer 功能测试、5 项补充测试、9 项模型组件测试和 1 项 Softmax 测试；另外 2 项官方内存测试由平台条件跳过。本次新增的多头因果自注意力官方测试通过。

2026-10-01（第二次同步）：README 中列出的已实现模块测试在 macOS / Python 3.12 下得到 **40 passed，2 skipped**。其中包含 3 项 BPE 测试、23 项官方 Tokenizer 功能测试、5 项补充测试、8 项模型组件测试和 1 项 Softmax 测试；另外 2 项官方内存测试由平台条件跳过。本次新增的 Softmax 与两项缩放点积注意力测试均通过。

2026-10-01（首次发布）：当时已有模块的测试结果为 **37 passed，2 skipped**。

当前模型组件、TransformerBlock、完整 TransformerLM 前向计算与交叉熵损失的官方测试均已通过。优化器、数据采样、梯度裁剪与 checkpoint 等接口仍待实现，模型训练和生成实验尚未完成。

## 后续作业

A2 Systems、A3 Scaling、A4 Data、A5 Alignment 尚未开始；本仓库暂不收录这些作业的空白起始副本。

## 更新记录模板

后续新增记录时注明：日期、完成内容、验证命令与结果、尚存问题和下次起点。
