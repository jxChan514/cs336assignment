"""Assignment 1 §6：先实现基础采样，再加入 top-p 和生成循环。"""

import torch


def sample_next_token(
    next_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """输入最后一个位置的 logits：(B, V)；返回采样的 ID：(B, 1)。"""
    # TODO：评分除以 temperature，再沿词表维度转成概率。
    # TODO：按概率采样一个 token ID，并返回。
    raise NotImplementedError("请先完成基础采样函数")


if __name__ == "__main__":
    # 调试输入：一条序列、4 个候选 token，方便观察每一步的数值和形状。
    torch.manual_seed(42)
    next_logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    temperature = 1.0
    # 在下面这一行左侧设置断点；F11 可进入你写的采样函数。
    next_token_id = sample_next_token(next_logits, temperature)
    print(f"采样结果: {next_token_id}，形状: {next_token_id.shape}")
