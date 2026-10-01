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
        self.register_buffer("freqs", freqs) # 注册为buffer，不参与训练

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






        

