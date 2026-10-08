import torch 
import torch.nn as nn

# ==================== 阅读形状时使用的符号 ====================
# B = batch_size：一次处理的序列条数；T = seq_len：每条序列的 token 数。
# D = d_model：每个 token 的特征数；H = num_heads：注意力头数。
# d = D // H：每个头的特征数；F = d_ff：前馈网络的中间维度。
# V = vocab_size：词表大小；L = max_seq_len：RoPE 角度表的容量。
# (...) 表示任意数量的前导维度，如 (B,) 或 (B, H)，不是某个固定维度。
# dim=-1 是最后一维，dim=-2 是倒数第二维，以此类推。
# @ 是矩阵乘法，作用于最后两个维度，前导维度按批次广播。
# * 是逐元素乘法；广播从右向左对齐，长度为 1 的维度可以扩展。
# 贯穿示例：B=2、T=5、D=12、H=3，因此 d=4。


# ==================== 1. 基础层：Linear / Embedding / RMSNorm ====================
class Linear(nn.Module):
    # 无偏置线性层：(..., in_features) -> (..., out_features)。
    def __init__(self,in_features,out_features,device=None,dtype=None):
        super().__init__()
        # 权重形状：(out_features, in_features)，每一行对应一个输出特征。
        weight=torch.empty(out_features,in_features,device=device,dtype=dtype)
        std=(2/(in_features+out_features))**0.5
        nn.init.trunc_normal_(weight,mean=0.0,std=std,a=-3*std,b=3*std)
        self.weight=nn.Parameter(weight)
    def forward(self,x:torch.Tensor):
        # 转置权重后：(..., in_features) @ (in_features, out_features)。
        # 例如 (B,T,12) @ (12,32) -> (B,T,32)，不会改变批次和序列长度。
        y=x@self.weight.transpose(0,1)
        return y

class Embedding(nn.Module):
    # 查表层：整数 ID (...,) -> 浮点向量 (..., embedding_dim)。
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # 权重形状：(V,D)。第 i 行就是 token ID=i 对应的可训练向量。
        # 行数由词表大小决定，与当前输入序列长度 T 无关。
        weight=torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype)
        nn.init.trunc_normal_(weight,mean=0.0,std=1,a=-3,b=3)
        self.weight=nn.Parameter(weight)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids：(B,T)，每个值是 [0,V) 中的整数；查表结果：(B,T,D)。
        # 例如 [[2,7,1]] 的形状为 (1,3)，查出 3 个 D 维向量，得到 (1,3,D)。
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    # 每个 token 独立沿最后的 D 个特征归一化，输入输出均为 (...,D)。
    def __init__(self, d_model, eps=1e-5,device=None,dtype=None):
        super().__init__()
        self.eps=eps
        # 可训练缩放系数：(D,)，所有 token 共享这 D 个参数。
        weight=torch.ones(d_model,device=device,dtype=dtype)
        self.weight=nn.Parameter(weight)
        self.dtype=dtype
    def forward(self,x):
        dtype=x.dtype
        # 用 float32 进行平方和均值计算，减少低精度计算的溢出风险。
        x=x.to(torch.float32)
        x_sq=x**2
        # x_sq：(B,T,D) -> x_mean：(B,T,1)。keepdim 保留最后一维用于广播。
        x_mean=x_sq.mean(dim=-1,keepdim=True)
        rms=torch.sqrt(x_mean+self.eps)
        # (B,T,D) / (B,T,1)：每个 token 的 D 个特征除以它自己的 rms。
        x_normed=x/rms
        # (B,T,D) * (D,) -> (B,T,D)，各特征再乘可训练缩放系数。
        output=x_normed*self.weight
        output=output.to(dtype=dtype)
        return output

# ==================== 2. 逐位置前馈网络：SwiGLU ====================
def silu(x:torch.Tensor):

    # 逐元素激活：x * sigmoid(x)，不改变形状，不混合不同 token。
    return x*torch.sigmoid(x)

class SwiGLU(nn.Module):
    # 输入输出：(...,D)。两路 D -> F，门控相乘后再 F -> D。
    def __init__(self, d_model, d_ff,device=None,dtype=None):
        super().__init__()
        # w1.weight、w3.weight：(F,D)；w2.weight：(D,F)。
        self.w1=Linear(in_features=d_model,out_features=d_ff,device=device,dtype=dtype)
        self.w2=Linear(in_features=d_ff,out_features=d_model,device=device,dtype=dtype)
        self.w3=Linear(in_features=d_model,out_features=d_ff,device=device,dtype=dtype)
    def forward(self,x):
        # 门控分支：(...,D) -> (...,F)，经过 SiLU。
        x_1=silu(self.w1(x))
        # 另一分支：(...,D) -> (...,F)。
        x_3=self.w3(x)
        # 两路逐元素相乘，仍为 (...,F)，这里不是矩阵乘法。
        hidden=x_1*x_3
        # (...,F) -> (...,D)，恢复模型维度，方便残差相加。
        return self.w2(hidden)

# ==================== 3. 旋转位置编码：RoPE ====================
class RotaryPositionalEmbedding(nn.Module):
    # 在多头注意力中 d_k=d，旋转每个头内相邻的特征对，不改变 token 顺序。
    def __init__(self,theta,d_k,max_seq_len,device=None):
        super().__init__()
        assert d_k%2==0
        self.theta=theta
        self.max_seq_len=max_seq_len
        self.device=device
        # i：(d_k//2,)，每两个相邻特征为一组，共有 d_k//2 组。
        i=torch.arange(start=0,end=d_k//2,dtype=torch.float32,device=device)
        # 每组的旋转频率：(d_k//2,)，不同组使用不同频率。
        freqs=1/(theta**(2*i/d_k))
        # (d_k//2,) -> (1,d_k//2)，让频率在所有位置上广播。
        freqs=freqs.reshape(1,-1)
        # 位置编号：(L,)，从 0 到 L-1；L 是容量，不是每次输入必须达到的长度。
        pos=torch.arange(start=0,end=max_seq_len,dtype=torch.float32,device=device)
        # (L,) -> (L,1)，让每个位置与所有频率相乘。
        pos=pos.reshape(-1,1)
        # 旋转角度表，形状为 (max_seq_len, d_k // 2)。
        # m_theta[m, j] = 位置 m × 第 j 对元素的旋转频率。
        # 例如 d_k=4、theta=10000 时，位置 3 对应的角度为 [3, 0.03] 弧度。
        m_theta=pos*freqs
        # 两张表均为 (L,d_k//2)，每一行对应一个位置的各组旋转角度。
        cos_table=torch.cos(m_theta)
        sin_table=torch.sin(m_theta)

        # buffer 随模块移动设备，但不参与训练；persistent=False 表示不写入 state_dict。
        self.register_buffer("cos_cached", cos_table, persistent=False)
        self.register_buffer("sin_cached", sin_table, persistent=False)
        self.register_buffer("freqs", freqs,persistent=False) # 注册为buffer，不参与训练

    # x：(...,T,d_k)。token_positions 存储位置整数，不是词表中的 token ID。
    # 位置的前导维度需要能与 x 广播；多头输入的具体对齐方式见下方示例。
    def forward(self,x,token_positions):
        # 查表在位置张量末尾增加一维：positions.shape + (d_k//2,)。
        # 例如 x=(B,H,T,d)，positions=(B,1,T)，查表结果=(B,1,T,d//2)。
        # 若 positions=(T,)，则查表结果=(T,d//2)，所有批次/头共享这套位置。
        cos=self.cos_cached[token_positions]
        sin=self.sin_cached[token_positions]
        # 每组角度供两个特征使用：[c0,c1] -> [c0,c0,c1,c1]。
        # (B,1,T,d//2) -> (B,1,T,d)，最后一维用 dim=-1 表示。
        cos = cos.repeat_interleave(2, dim=-1)
        sin=sin.repeat_interleave(2,dim=-1)
        # 每对元素的第一项：[a,c,...]，形状 (...,T,d_k//2)。
        x_even = x[..., 0::2]

        # 每对元素的第二项：[b,d,...]，形状与 x_even 相同。
        x_odd = x[..., 1::2]

        # stack 后：(...,T,d_k//2,2)，每组为 [-b,a]、[-d,c] 等。
        # flatten(-2) 合并最后两维，恢复 (...,T,d_k)：[-b,a,-d,c,...]。
        # x_rotated 是辅助向量；结合下一行的 cos/sin 才得到最终位置旋转。
        x_rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
        # 多头示例：(B,H,T,d) * (B,1,T,d)，角度沿头数 H 广播。
        # 每对 [a,b] 变为 [a*cos-b*sin, b*cos+a*sin]，输出与 x 形状相同。
        output=x*cos+x_rotated*sin
        return output.to(dtype=x.dtype)

# ==================== 4. Softmax 与缩放点积注意力 ====================
def softmax(x:torch.Tensor,dim:int):
    # 沿 dim 指定的轴归一化，该轴上的概率和为 1；输出形状与 x 相同。
    # 例如 x=(B,H,T,T)、dim=-1，max_value/exp_sum 均为 (B,H,T,1)。
    max_value=torch.max(x,dim=dim,keepdim=True).values
    # 减去最大值不会改变 softmax 结果，且避免 exp(很大的正数) 溢出。
    x=x-max_value
    x_exp=torch.exp(x)
    exp_sum=torch.sum(x_exp,dim=dim,keepdim=True)
    output=x_exp/exp_sum
    return output

def scaled_dot_product_attention(query:torch.Tensor,key:torch.Tensor,value:torch.Tensor,mask=None):
    # 通用形状：Q=(...,Tq,d_k)，K=(...,Tk,d_k)，V=(...,Tk,d_v)。
    # 自注意力中 Tq=Tk=T；你的多头版本中前导维度是 (B,H)，d_k=d_v=d。
    # K 转置为 (...,d_k,Tk)，Q @ K^T 得到 (...,Tq,Tk)。
    # 分数矩阵的行是查询位置，列是被关注的位置；这里不在不同头之间做乘法。
    relevant_score=query@key.transpose(dim0=-1,dim1=-2)
    d_k=query.shape[-1]
    scores=relevant_score/(d_k**(0.5))
    if mask is not None:
        # mask=True 表示允许关注；~mask=True 的位置替换成 -inf。
        # (Tq,Tk) 的 mask 可以广播到所有批次和头，softmax 后被屏蔽位置权重为 0。
        scores = scores.masked_fill(~mask, float("-inf"))
    # 沿最后的 Tk 维归一化：每个查询对所有可见 key 的权重和为 1。
    scores_softmax=softmax(scores,dim=-1)
    # (...,Tq,Tk) @ (...,Tk,d_v) -> (...,Tq,d_v)，对 V 做加权求和。
    output=scores_softmax@value
    return output

# ==================== 5. 因果多头自注意力 ====================
class MultiHeadSelfAttention(nn.Module):
    # 输入输出均为 (...,T,D)，多头只改变中间表示，最后会合回 D 维。
    def __init__(self, d_model, num_heads, device=None, dtype=None,theta=None,max_seq_len=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        # 四个投影的权重均为 (D,D)。先一次投影所有头，再拆分最后的 D 维。
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is None and max_seq_len is None:
            self.rope=None
        elif theta is not None and max_seq_len is not None:
            # RoPE 在每个头内旋转，所以 d_k 使用 d_head，而不是整个 d_model。
            self.rope=RotaryPositionalEmbedding(theta=theta,d_k=self.d_head,max_seq_len=max_seq_len,device=device)
        else:
            raise ValueError("theta and max_seq_len must be provided together, or both None")
        

    def forward(self, x,token_positions=None):
        # 投影：(..., seq_len, d_model)。省略号代表前面的批次维度。
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # unflatten：(...,T,D) -> (...,T,H,d)；transpose：交换 T 和 H。
        # 示例：(2,5,12) -> (2,5,3,4) -> (2,3,5,4)。
        # 拆分的是每个 token 的特征，所有头都仍然包含完整的 T 个位置。
        head_shape = (self.num_heads, self.d_head)
        q = q.unflatten(-1, head_shape).transpose(-3, -2)
        k = k.unflatten(-1, head_shape).transpose(-3, -2)
        v = v.unflatten(-1, head_shape).transpose(-3, -2)
        

        # x 的倒数第二维始终是当前序列长度 T。
        seq_len = x.shape[-2]

        if self.rope is not None:
            if token_positions is None:
                # 默认位置：(T,)，整数 [0,1,...,T-1]，与 x 位于同一设备。
                token_positions=torch.arange(0,seq_len,device=x.device)
            # 为各头共享位置留出长度为 1 的维度：(B,T) -> (B,1,T)。
            # 默认的一维位置同样可用：(T,) -> (1,T)，生成的角度可广播到所有批次/头。
            rope_positions = token_positions.unsqueeze(-2)
            # 只旋转 Q、K，形状不变；V 保留用于加权求和的内容。
            q=self.rope(q,rope_positions)
            k=self.rope(k,rope_positions)
        
        # (T,T) 下三角布尔矩阵。第 i 行第 j 列为 True 当且仅当 j<=i。
        # 包含对角线：可以关注自己，但不能提前看到未来的 token。
        causal_mask = torch.ones(
            seq_len, seq_len, device=x.device, dtype=torch.bool
        ).tril()

        # 各头的注意力结果：(..., num_heads, seq_len, d_head)。
        head_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        # (...,H,T,d) -> (...,T,H,d) -> (...,T,D)，必须先交换再合并。
        # 示例：(2,3,5,4) -> (2,5,3,4) -> (2,5,12)。
        output=head_output.transpose(-3,-2).flatten(-2)
        # 对拼接后的各头特征进行线性混合，仍为 (...,T,D)。残差连接由 Block 完成。
        output=self.output_proj(output)
        return output

# ==================== 6. Pre-Norm Transformer Block ====================
class TransformerBlock(nn.Module):
    # 两个阶段：h=x+Attention(ln1(x))，y=h+FFN(ln2(h))。
    # 各阶段的输入/输出均为 (...,T,D)，因此可以逐元素进行残差相加。
    def __init__(self,d_model,num_heads,d_ff,theta,max_seq_len,device=None,dtype=None):
        super().__init__()
        self.ln1=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.attn=MultiHeadSelfAttention(d_model=d_model,
                                         num_heads=num_heads,
                                         device=device,
                                         dtype=dtype,
                                         theta=theta,
                                         max_seq_len=max_seq_len)
        # ln1 与 ln2 是独立模块，各自拥有 (D,) 的缩放权重。
        self.ln2=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.ffn=SwiGLU(d_model=d_model,d_ff=d_ff,device=device,dtype=dtype)
    def forward(self,x):
        # 第一条残差支路保留原始 x；这里只是别名，不会复制张量。
        # 后面没有原地修改 x，因此可直接用它与注意力结果相加。
        x_copy=x
        # (...,T,D) -> (...,T,D)：先归一化，再通过注意力混合不同位置的信息。
        x_norm=self.ln1(x)
        x_attn=self.attn(x_norm)
        hidden=x_copy+x_attn
        # 第二条残差保留第一阶段的结果 hidden，而不是最初的 x。
        hidden_copy=hidden
        hidden_norm=self.ln2(hidden)
        # FFN 内部先扩展到 (...,T,F)，再恢复到 (...,T,D)。
        hidden_output=self.ffn(hidden_norm)
        output=hidden_output+hidden_copy
        return output

# ==================== 7. 完整语言模型：TransformerLM ====================
class TransformerLM(nn.Module):
    # 输入是整数 token ID：(B,T)；输出是浮点 logits：(B,T,V)。
    # 示例：V=10000、D=12 时，(2,5) -> (2,5,12) -> (2,5,10000)。
    def __init__(self,vocab_size,
                 context_length,
                 d_model,d_ff,
                 num_layers,
                 num_heads,
                 rope_theta,
                 device=None,dtype=None):
        super().__init__()
        # Embedding 权重：(V,D)，将每个整数 ID 替换为一个 D 维向量。
        self.token_embeddings=Embedding(num_embeddings=vocab_size,embedding_dim=d_model,device=device,dtype=dtype)
        # ModuleList 注册各层，使其参数能被优化器、state_dict 和 .to(device) 找到。
        # 每次循环都创建一个新 Block，各层权重独立；每层都保持 (...,T,D) 的形状。
        # context_length 用于各层 RoPE 的缓存容量；当前输入 T 可以比它小。
        self.layers=nn.ModuleList([TransformerBlock(d_model=d_model,num_heads=num_heads,d_ff=d_ff,theta=rope_theta,max_seq_len=context_length,device=device,dtype=dtype) for _ in range(num_layers)])
        self.ln_final=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        # LM Head 权重：(V,D)，与 Embedding 的权重形状相同，但这里是独立参数。
        self.lm_head=Linear(in_features=d_model,out_features=vocab_size,device=device,dtype=dtype)
    def forward(self,x):
        # 此处 x 最初是 (B,T) 整数 ID；查表后重新赋值为 (B,T,D) 浮点向量。
        x=self.token_embeddings(x)
        for block in self.layers:
            # 上一层的输出作为下一层输入，每经过一层形状仍为 (B,T,D)。
            x=block(x)
        # 最终归一化不改变形状：(B,T,D)。
        x_norm=self.ln_final(x)
        # (B,T,D) @ (D,V) -> (B,T,V)，最后一维对应词表中的所有候选 token。
        output=self.lm_head(x_norm)
        # output[b,t,:] 是读到位置 t 后对下一个 token 的评分，不是概率。
        # 返回 logits，不在这里做 softmax；训练时交给交叉熵损失处理。
        return output

def cross_entropy(logits,targets):
    # logits：(B,T,V)，每个预测位置对词表中 V 个候选 token 的评分。
    # targets：(B,T)，每个位置应当预测出的下一个 token 的 ID。
    # 也支持 logits：(N,V)、targets：(N,)；下面始终沿最后的词表维度运算。

    # 每个位置取词表中的最大评分：(B,T,V) -> (B,T,1)。
    # 保留最后一维，便于将这个最大值广播到该位置的全部 V 个评分。
    max_val=torch.max(logits,dim=-1,keepdim=True).values
    # 每个位置的评分同时减去自己的最大值，形状仍为 (B,T,V)。
    # 最大评分变成 0，可避免后面的 exp 溢出；softmax 概率保持不变。
    logits=logits-max_val

    # 对平移后的每个评分取指数，仍为 (B,T,V)。
    logits_exp=torch.exp(logits)
    # 在词表维度求和：(B,T,V) -> (B,T)，得到每个位置的归一化分母。
    logits_sum=torch.sum(logits_exp,dim=-1,keepdim=False)
    # 分母取自然对数，形状为 (B,T)。
    log_partition=torch.log(logits_sum)

    # (B,T) -> (B,T,1)：每个位置只查找一个正确答案的评分。
    target_indices = targets.unsqueeze(-1)
    # gather 沿词表维度按 targets 查分数，得到 (B,T,1)。
    # squeeze(-1) 去掉查找用的单元素维度，得到 (B,T)。
    # 取出的是减去最大值后的评分，与上面的 log_partition 使用同一套评分。
    target_logits=torch.gather(logits,dim=-1,index=target_indices).squeeze(-1)

    # 每个位置的交叉熵：-log(正确 token 的预测概率)。
    # loss：(B,T)，其中 loss[b,t] 衡量第 b 条序列、第 t 个位置的预测误差。
    loss=log_partition-target_logits
    # mean() 不指定维度：对 loss 中全部 B*T 个位置求平均，返回标量，形状为 ()。
    # 即 sum(loss[b,t]) / (B*T)，同时平均 batch 维和序列维；词表维度已被求和消去。
    # 例如 B=2、T=32：最终得到这 64 个预测位置的平均 loss，每个位置权重相同。
    return loss.mean()
