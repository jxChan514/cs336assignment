# 更新日志

本仓库持续更新；仅把实际实现和验证过的内容标为完成。

## 2026-10-04

- 新增 `TransformerLM`，组合 token embedding、多层 TransformerBlock、最终 RMSNorm 与语言模型输出投影，返回每个位置的词表 logits。
- 在官方测试适配器中接入语言模型实现与权重加载。
- TransformerLM 正常输入与截短输入的 2 项官方测试均通过；`test_model.py` 全部 13 项测试通过，已有实现回归测试合计：45 passed，2 skipped（macOS / Python 3.12）。
- 更新 README 和学习进度；损失函数、优化器、数据采样、训练与生成流程仍待实现。

## 2026-10-03

- 为 `MultiHeadSelfAttention` 接入可选 RoPE，支持传入 token positions，并在省略时生成默认位置。
- 将 RoPE 的频率缓存设为非持久化 buffer。
- 新增 `TransformerBlock`，组合 RMSNorm、带 RoPE 的多头注意力、SwiGLU 与残差连接。
- 接入对应官方测试适配器；新增的 2 项官方测试通过，已有实现回归测试合计：43 passed，2 skipped（macOS / Python 3.12）。
- 更新 README 和学习进度；完整语言模型、优化器与训练流程仍待实现。

## 2026-10-02

- 新增 `MultiHeadSelfAttention`：Q/K/V 投影、拆头、因果 mask、合并各头与输出投影。
- 在官方测试适配器中接入无 RoPE 的多头自注意力实现。
- 新增功能的官方测试通过，已有实现回归测试合计：41 passed，2 skipped（macOS / Python 3.12）。
- 更新 README 和学习进度；带 RoPE 的多头注意力、完整 Transformer 与训练流程仍待实现。

## 2026-10-01 · 第二次同步

- 新增数值稳定的 Softmax 与支持 mask 的缩放点积注意力实现。
- 在官方测试适配器中接入 Softmax 与缩放点积注意力。
- 新增功能的 3 项官方测试通过，已有实现回归测试合计：40 passed，2 skipped（macOS / Python 3.12）。
- 更新 README 和学习进度；多头注意力、完整 Transformer 与训练流程仍待实现。

## 2026-10-01 · 首次发布

- 首次整理并发布 Assignment 1 的现有学习实现。
- 收录字节级 BPE 训练、Tokenizer 编解码、特殊 token 处理及 JSON 文件接口。
- 收录 Linear、Embedding、RMSNorm、SiLU、SwiGLU 和 RoPE。
- 保留官方测试、测试夹具、依赖锁定文件、作业说明和许可证，加入 5 项 Tokenizer 补充测试。
- 已实现模块在 macOS / Python 3.12 下验证：37 passed，2 skipped。
- Attention、完整 Transformer、优化器及训练流程仍待实现。
