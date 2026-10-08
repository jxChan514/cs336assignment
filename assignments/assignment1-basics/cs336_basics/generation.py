"""Assignment 1 §6：先实现基础采样，再加入 top-p 和生成循环。"""

import torch


def sample_next_token(
    next_logits: torch.Tensor,
    temperature: float = 1.0,
    top_p:float=1.0
) -> torch.Tensor:
    """输入最后一个位置的 logits：(B, V)；返回采样的 ID：(B, 1)。"""
    socres=next_logits/temperature
    prob=torch.softmax(socres,dim=-1)
    values,indices=torch.sort(prob,dim=-1,descending=True)
    cumulative_probs=torch.cumsum(values,dim=-1)
    mask = cumulative_probs < top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = True

    filtered_probs=values.masked_fill(~mask, 0.0)
    sum_probs=filtered_probs.sum(dim=-1,keepdim=True)
    filtered_probs=filtered_probs/sum_probs
    sampled_positions = torch.multinomial(filtered_probs, num_samples=1)
    next_token_ids=torch.gather(input=indices,dim=-1,index=sampled_positions)
    return next_token_ids

def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        generate_ids=input_ids.clone()
        for _ in range(max_new_tokens):
            inputs=generate_ids[:,-context_length:]
            logits=model(inputs)
            next_logits=logits[:,-1,:]
            next_token_id=sample_next_token(next_logits=next_logits,temperature=temperature,top_p=top_p)
            tokens=torch.cat([generate_ids,next_token_id],dim=1)
            generate_ids=tokens
            if(next_token_id.item()==eos_token_id):
                break
            


    




if __name__ == "__main__":
    # 调试输入：一条序列、4 个候选 token，方便观察每一步的数值和形状。
    torch.manual_seed(42)
    next_logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    temperature = 1.0
    # 在下面这一行左侧设置断点；F11 可进入你写的采样函数。
    next_token_id = sample_next_token(next_logits, temperature,top_p=0.8)
    print(f"采样结果: {next_token_id}，形状: {next_token_id.shape}")

    # 生成循环的调试入口：完整输入有 5 个 token，模型上下文最多为 4。
    from cs336_basics.model import TransformerLM

    debug_model = TransformerLM(
        vocab_size=16,
        context_length=4,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_heads=2,
        rope_theta=10000,
        device="cpu",
        dtype=torch.float32,
    )
    debug_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    # 在 generate() 的 generate_ids = input_ids.clone() 这一行设置断点。
    generation_result = generate(
        model=debug_model,
        input_ids=debug_input_ids,
        max_new_tokens=2,
        context_length=4,
    )
    # 目前函数暂时返回 next_logits；先通过断点检查形状，再继续实现生成。

