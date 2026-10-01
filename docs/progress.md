# 学习进度

最后更新：2026-10-01。状态以当前公开源码和已执行测试为准。

## Assignment 1 · Basics

- [x] 字节级 BPE 训练：预分词、pair 统计、合并与特殊 token 边界。
- [x] Tokenizer：encode / decode、特殊 token、按段 encode_iterable。
- [x] Tokenizer 文件接口：JSON 保存与加载，保留任意字节。
- [x] Tokenizer 边界测试：重叠特殊 token、非法 UTF-8、稀疏 ID、文件往返、惰性迭代。
- [x] Linear、Embedding、RMSNorm、SiLU、SwiGLU、RoPE。
- [x] 数值稳定的 Softmax，相关测试通过。
- [x] Scaled dot-product attention，标准输入与四维输入测试通过。
- [ ] Multi-head self-attention。
- [ ] Transformer block 与完整语言模型。
- [ ] Cross-entropy、数据采样、梯度裁剪。
- [ ] AdamW、学习率调度、checkpoint。
- [ ] 大型语料 BPE、模型训练、生成与实验报告。
- [ ] Linux 下官方 tokenizer 内存限制测试。

## 验证记录

2026-10-01（第二次同步）：README 中列出的已实现模块测试在 macOS / Python 3.12 下得到 **40 passed，2 skipped**。其中包含 3 项 BPE 测试、23 项官方 Tokenizer 功能测试、5 项补充测试、8 项模型组件测试和 1 项 Softmax 测试；另外 2 项官方内存测试由平台条件跳过。本次新增的 Softmax 与两项缩放点积注意力测试均通过。

2026-10-01（首次发布）：当时已有模块的测试结果为 **37 passed，2 skipped**。

额外检查多头注意力（含 RoPE）测试时，当前适配接口因尚未实现而触发 `NotImplementedError`。RoPE 独立组件测试已通过；这不代表多头注意力已完成。

## 后续作业

A2 Systems、A3 Scaling、A4 Data、A5 Alignment 尚未开始；本仓库暂不收录这些作业的空白起始副本。

## 更新记录模板

后续新增记录时注明：日期、完成内容、验证命令与结果、尚存问题和下次起点。
