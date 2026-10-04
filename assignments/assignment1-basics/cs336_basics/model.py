import torch 
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self,in_features,out_features,device=None,dtype=None):
        super().__init__()
        weight=torch.empty(out_features,in_features,device=device,dtype=dtype)
        std=(2/(in_features+out_features))**0.5
        nn.init.trunc_normal_(weight,mean=0.0,std=std,a=-3*std,b=3*std)
        self.weight=nn.Parameter(weight)
    def forward(self,x:torch.Tensor):
        y=x@self.weight.transpose(0,1)
        return y

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        weight=torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype)
        nn.init.trunc_normal_(weight,mean=0.0,std=1,a=-3,b=3)
        self.weight=nn.Parameter(weight)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5,device=None,dtype=None):
        super().__init__()
        self.eps=eps
        weight=torch.ones(d_model,device=device,dtype=dtype)
        self.weight=nn.Parameter(weight)
        self.dtype=dtype
    def forward(self,x):
        dtype=x.dtype
        x=x.to(torch.float32)
        x_sq=x**2
        x_mean=x_sq.mean(dim=-1,keepdim=True)
        rms=torch.sqrt(x_mean+self.eps)
        x_normed=x/rms
        output=x_normed*self.weight
        output=output.to(dtype=dtype)
        return output

def silu(x:torch.Tensor):

    return x*torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff,device=None,dtype=None):
        super().__init__()
        self.w1=Linear(in_features=d_model,out_features=d_ff,device=device,dtype=dtype)
        self.w2=Linear(in_features=d_ff,out_features=d_model,device=device,dtype=dtype)
        self.w3=Linear(in_features=d_model,out_features=d_ff,device=device,dtype=dtype)
    def forward(self,x):
        x_1=silu(self.w1(x))
        x_3=self.w3(x)
        hidden=x_1*x_3
        return self.w2(hidden)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta,d_k,max_seq_len,device=None):
        super().__init__()
        assert d_k%2==0
        self.theta=theta
        self.max_seq_len=max_seq_len
        self.device=device
        #freqs:
        i=torch.arange(start=0,end=d_k//2,dtype=torch.float32,device=device)
        freqs=1/(theta**(2*i/d_k))
        freqs=freqs.reshape(1,-1)
        pos=torch.arange(start=0,end=max_seq_len,dtype=torch.float32,device=device)
        pos=pos.reshape(-1,1)
        # 旋转角度表，形状为 (max_seq_len, d_k // 2)。
        # m_theta[m, j] = 位置 m × 第 j 对元素的旋转频率。
        # 例如 d_k=4、theta=10000 时，位置 3 对应的角度为 [3, 0.03] 弧度。
        m_theta=pos*freqs
        cos_table=torch.cos(m_theta)
        sin_table=torch.sin(m_theta)

        self.register_buffer("cos_cached", cos_table, persistent=False)
        self.register_buffer("sin_cached", sin_table, persistent=False)
        self.register_buffer("freqs", freqs,persistent=False) # 注册为buffer，不参与训练

    #x：(batch_size, seq_len, d_k),token_positions：(batch_size, seq_len)    
    def forward(self,x,token_positions):
        cos=self.cos_cached[token_positions]
        sin=self.sin_cached[token_positions]
        cos = cos.repeat_interleave(2, dim=-1) # 在dim=1（最后一维）每个元素复制2次
        sin=sin.repeat_interleave(2,dim=-1)
        # 每对元素的第一项：[a, c, ...]
        x_even = x[..., 0::2]

        # 每对元素的第二项：[b, d, ...]
        x_odd = x[..., 1::2]

        # 先得到 [[-b, a], [-d, c], ...]，再展开为 [-b, a, -d, c, ...]。
        x_rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
        output=x*cos+x_rotated*sin
        return output.to(dtype=x.dtype)

def softmax(x:torch.Tensor,dim:int):
    max_value=torch.max(x,dim=dim,keepdim=True).values
    x=x-max_value
    x_exp=torch.exp(x)
    exp_sum=torch.sum(x_exp,dim=dim,keepdim=True)
    output=x_exp/exp_sum
    return output

def scaled_dot_product_attention(query:torch.Tensor,key:torch.Tensor,value:torch.Tensor,mask=None):
    relevant_score=query@key.transpose(dim0=-1,dim1=-2)
    d_k=query.shape[-1]
    scores=relevant_score/(d_k**(0.5))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    scores_softmax=softmax(scores,dim=-1)
    output=scores_softmax@value
    return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None,theta=None,max_seq_len=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is None and max_seq_len is None:
            self.rope=None
        elif theta is not None and max_seq_len is not None:
            self.rope=RotaryPositionalEmbedding(theta=theta,d_k=self.d_head,max_seq_len=max_seq_len,device=device)
        else:
            raise ValueError("theta and max_seq_len must be provided together, or both None")
        

    def forward(self, x,token_positions=None):
        # 投影：(..., seq_len, d_model)。省略号代表前面的批次维度。
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 拆头并交换维度：(..., num_heads, seq_len, d_head)。
        head_shape = (self.num_heads, self.d_head)
        q = q.unflatten(-1, head_shape).transpose(-3, -2)
        k = k.unflatten(-1, head_shape).transpose(-3, -2)
        v = v.unflatten(-1, head_shape).transpose(-3, -2)
        

        # 因果 mask：(seq_len, seq_len)，True 表示允许关注自己或之前的位置。
        seq_len = x.shape[-2]

        if self.rope is not None:
            if token_positions is None:
                token_positions=torch.arange(0,seq_len,device=x.device)
            rope_positions = token_positions.unsqueeze(-2)
            q=self.rope(q,rope_positions)
            k=self.rope(k,rope_positions)
        
        causal_mask = torch.ones(
            seq_len, seq_len, device=x.device, dtype=torch.bool
        ).tril()

        # 各头的注意力结果：(..., num_heads, seq_len, d_head)。
        head_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        # 下一步：合并各头，再经过 output_proj 并返回结果。
        output=head_output.transpose(-3,-2).flatten(-2)
        output=self.output_proj(output)
        return output
class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,theta,max_seq_len,device=None,dtype=None):
        super().__init__()
        self.ln1=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.attn=MultiHeadSelfAttention(d_model=d_model,
                                         num_heads=num_heads,
                                         device=device,
                                         dtype=dtype,
                                         theta=theta,
                                         max_seq_len=max_seq_len)
        self.ln2=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.ffn=SwiGLU(d_model=d_model,d_ff=d_ff,device=device,dtype=dtype)
    def forward(self,x):
        x_copy=x
        x_norm=self.ln1(x)
        x_attn=self.attn(x_norm)
        hidden=x_copy+x_attn
        hidden_copy=hidden
        hidden_norm=self.ln2(hidden)
        hidden_output=self.ffn(hidden_norm)
        output=hidden_output+hidden_copy
        return output
class TransformerLM(nn.Module):
    def __init__(self,vocab_size,
                 context_length,
                 d_model,d_ff,
                 num_layers,
                 num_heads,
                 rope_theta,
                 device=None,dtype=None):
        super().__init__()
        self.token_embeddings=Embedding(num_embeddings=vocab_size,embedding_dim=d_model,device=device,dtype=dtype)
        self.layers=nn.ModuleList([TransformerBlock(d_model=d_model,num_heads=num_heads,d_ff=d_ff,theta=rope_theta,max_seq_len=context_length,device=device,dtype=dtype) for _ in range(num_layers)])
        self.ln_final=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.lm_head=Linear(in_features=d_model,out_features=vocab_size,device=device,dtype=dtype)
    def forward(self,x):
        x=self.token_embeddings(x)
        for block in self.layers:
            x=block(x)
        x_norm=self.ln_final(x)
        output=self.lm_head(x_norm)
        return output