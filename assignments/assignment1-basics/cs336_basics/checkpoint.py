import os
from typing import BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike[str] | BinaryIO,
) -> None:
    """保存模型、优化器和已完成的训练步数，支持路径或二进制文件对象。"""
    checkpoint = {
        # 模型状态：Embedding、Linear、RMSNorm 等模块的权重。
        "model": model.state_dict(),
        # 优化器状态：参数组配置，以及每个参数的 AdamW 历史记录 m、v、t。
        "optimizer": optimizer.state_dict(),
        # 约定为已经完成的更新次数；训练循环恢复后从下一次更新继续。
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike[str] | BinaryIO,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """将存档装回已有模型和优化器，并返回保存的训练步数。"""
    # 先在 CPU 读取，允许读取原先在其他设备上保存的存档。
    # load_state_dict 会将状态装入已有模型和优化器所使用的设备。
    # 存档只包含状态字典、张量和基础类型，因此可以使用 weights_only=True。
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
