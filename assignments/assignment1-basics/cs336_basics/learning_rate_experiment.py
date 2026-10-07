import math
from typing import Callable, Optional

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data

                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss

# 固定初始权重，让不同学习率的比较更公平。
torch.manual_seed(42)
initial_weights = 5 * torch.randn((10, 10))

for lr in [10, 100, 1000]:
    # 每组实验重新创建参数和优化器，从相同起点开始。
    weights = torch.nn.Parameter(initial_weights.clone())
    opt = SGD([weights], lr=lr)

    print(f"\n学习率：{lr}")

    for t in range(10):
        opt.zero_grad()                 # 清空上一次的梯度
        loss = (weights**2).mean()       # 当前损失
        print(f"第 {t + 1} 次：loss = {loss.item():.6g}")
        loss.backward()                 # 计算梯度，保存到 weights.grad
        opt.step()                      # 根据梯度更新 weightss
    