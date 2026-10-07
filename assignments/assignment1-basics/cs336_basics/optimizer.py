import math  # sqrt：计算 AdamW 的偏差修正系数
from collections.abc import Callable, Iterable  # closure 和参数集合的类型注解

import torch  # 张量运算，以及用 zeros_like 初始化 m、v
from torch.optim import Optimizer  # 自定义 AdamW 需要继承的父类

class AdamW(Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults={
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay
        }
        super().__init__(params=params,defaults=defaults)
    def step(self, closure=None):
        # closure 在正常梯度模式下执行，参数更新则放在 no_grad 中。
        loss = None if closure is None else closure()
        with torch.no_grad():
            for group in self.param_groups:
                params = group["params"]  # 当前组的参数张量列表
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                for p in params:
                    if p.grad is None:
                        continue
                    grad = p.grad

                    # 每个参数张量有独立的历史记录；m、v 的形状与 p 相同。
                    state = self.state[p]
                    if not state:
                        state["t"] = 0
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                    state["t"] += 1
                    t = state["t"]

                    m_new = beta1 * state["m"] + (1 - beta1) * grad
                    v_new = beta2 * state["v"] + (1 - beta2) * grad**2
                    state["m"] = m_new
                    state["v"] = v_new

                    # 普通数值的平方根用 math.sqrt；修正系数必须包含 t 次方。
                    step_size = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                    # 原地更新当前参数 p，不能修改参数列表 params。
                    p *= 1 - lr * weight_decay
                    p -= step_size * m_new / (torch.sqrt(v_new) + eps)
        return loss

def get_lr_cosine_schedule(it,max_learning_rate,min_learning_rate,warmup_iters,cosine_cycle_iters):
    if it<warmup_iters:
        learning_rate_t=it/warmup_iters*max_learning_rate
    elif warmup_iters<=it<=cosine_cycle_iters:
        learning_rate_t=min_learning_rate+0.5*(max_learning_rate-min_learning_rate)*(1+math.cos((it-warmup_iters)*torch.pi/(cosine_cycle_iters-warmup_iters)))
    else:
        learning_rate_t=min_learning_rate



    return learning_rate_t

def gradient_clipping(parameters,max_l2_norm):
    parameters=list(parameters)
    grad_sum=0
    for p in parameters:
        if p.grad is None:
            continue
        else:
            grad_sum+=torch.sum(p.grad**2)
    L2_norm=math.sqrt(grad_sum)
    if L2_norm>max_l2_norm:
        c=max_l2_norm/(L2_norm+1e-6)
        for p in parameters:
            if p.grad is None:
                continue
            else:
                p.grad*=c
    
        
            
    
