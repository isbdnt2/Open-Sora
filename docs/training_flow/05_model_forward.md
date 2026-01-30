# 05. 模型前向传播 (Model Forward)

## 流程图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              MMDiT 前向传播流程                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          1. 输入投影                                         │   │
│   ├─────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                             │   │
│   │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │   │
│   │   │   img (x_t) │     │     txt     │     │   cond      │                   │   │
│   │   │ [B, N, 64]  │     │[B, L, 4096] │     │ [B, N, 68]  │                   │   │
│   │   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                   │   │
│   │          │                   │                   │                          │   │
│   │          ▼                   ▼                   ▼                          │   │
│   │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │   │
│   │   │   img_in    │     │   txt_in    │     │  cond_in    │                   │   │
│   │   │Linear(64,D) │     │Linear(4096,D)│    │Linear(68,D) │                   │   │
│   │   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                   │   │
│   │          │                   │                   │                          │   │
│   │          ▼                   │                   │                          │   │
│   │   ┌─────────────┐            │                   │                          │   │
│   │   │ img + cond  │◄───────────┘───────────────────┘                          │   │
│   │   │ [B, N, D]   │                                                           │   │
│   │   └──────┬──────┘                                                           │   │
│   │          │                                                                  │   │
│   │   D = hidden_size = 3072                                                    │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          2. 时间与条件嵌入                                   │   │
│   ├─────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                             │   │
│   │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │   │
│   │   │  timesteps  │     │   y_vec     │     │  guidance   │                   │   │
│   │   │    [B]      │     │  [B, 768]   │     │    [B]      │                   │   │
│   │   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                   │   │
│   │          │                   │                   │                          │   │
│   │          ▼                   ▼                   ▼                          │   │
│   │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │   │
│   │   │ TimestepEmb │     │   vec_in    │     │ GuidanceEmb │                   │   │
│   │   │  Sinusoidal │     │ MLP(768,D)  │     │  Sinusoidal │                   │   │
│   │   │   + MLP     │     │             │     │   + MLP     │                   │   │
│   │   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                   │   │
│   │          │                   │                   │                          │   │
│   │          └───────────────────┴───────────────────┘                          │   │
│   │                              │                                              │   │
│   │                              ▼                                              │   │
│   │                       ┌─────────────┐                                       │   │
│   │                       │  vec (sum)  │                                       │   │
│   │                       │  [B, D]     │                                       │   │
│   │                       └─────────────┘                                       │   │
│   │                              │                                              │   │
│   │          vec = time_emb + clip_emb + guidance_emb                           │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          3. 3D RoPE 位置编码                                 │   │
│   ├─────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                             │   │
│   │   根据 img_ids 和 txt_ids 计算 RoPE:                                        │   │
│   │                                                                             │   │
│   │   ┌─────────────────────────────────────────────────────────────────────┐   │   │
│   │   │   axes_dim = [16, 24, 24]  # T, H, W 各轴维度                       │   │   │
│   │   │   theta = 10000            # RoPE base frequency                    │   │   │
│   │   │                                                                     │   │   │
│   │   │   img_ids: [B, N, 3]  →  img_pe: [B, N, 1, 64]                      │   │   │
│   │   │   txt_ids: [B, L, 3]  →  txt_pe: [B, L, 1, 64]                      │   │   │
│   │   │                                                                     │   │   │
│   │   │   每个位置索引经过正弦/余弦编码                                     │   │   │
│   │   │   concat(sin(pos×freq), cos(pos×freq))                              │   │   │
│   │   │                                                                     │   │   │
│   │   └─────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          4. DoubleStreamBlock (×19)                          │   │
│   ├─────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                             │   │
│   │   双流处理: img 和 txt 各自有独立的 FFN，但共享 Attention                   │   │
│   │                                                                             │   │
│   │   ┌──────────────────────────────────────────────────────────────────────┐  │   │
│   │   │                                                                      │  │   │
│   │   │  img [B,N,D]                              txt [B,L,D]                │  │   │
│   │   │      │                                        │                      │  │   │
│   │   │      ▼                                        ▼                      │  │   │
│   │   │  ┌─────────┐                              ┌─────────┐                │  │   │
│   │   │  │ AdaLN   │◄────── vec ─────────────────►│ AdaLN   │                │  │   │
│   │   │  └────┬────┘                              └────┬────┘                │  │   │
│   │   │       │                                        │                     │  │   │
│   │   │       ▼                                        ▼                     │  │   │
│   │   │  ┌─────────┐                              ┌─────────┐                │  │   │
│   │   │  │  Q,K,V  │                              │  Q,K,V  │                │  │   │
│   │   │  │ img_attn│                              │ txt_attn│                │  │   │
│   │   │  └────┬────┘                              └────┬────┘                │  │   │
│   │   │       │                                        │                     │  │   │
│   │   │       └────────────────┬───────────────────────┘                     │  │   │
│   │   │                        │                                             │  │   │
│   │   │                        ▼                                             │  │   │
│   │   │              ┌────────────────────┐                                  │  │   │
│   │   │              │   Joint Attention  │                                  │  │   │
│   │   │              │ Q: [img_q; txt_q]  │                                  │  │   │
│   │   │              │ K: [img_k; txt_k]  │                                  │  │   │
│   │   │              │ V: [img_v; txt_v]  │                                  │  │   │
│   │   │              │   + RoPE           │                                  │  │   │
│   │   │              └────────────────────┘                                  │  │   │
│   │   │                        │                                             │  │   │
│   │   │           ┌────────────┴────────────┐                                │  │   │
│   │   │           ▼                         ▼                                │  │   │
│   │   │   ┌─────────────┐           ┌─────────────┐                          │  │   │
│   │   │   │ img_out     │           │ txt_out     │                          │  │   │
│   │   │   │ + gate×proj │           │ + gate×proj │                          │  │   │
│   │   │   └──────┬──────┘           └──────┬──────┘                          │  │   │
│   │   │          │                         │                                 │  │   │
│   │   │          ▼                         ▼                                 │  │   │
│   │   │   ┌─────────────┐           ┌─────────────┐                          │  │   │
│   │   │   │ img + FFN   │           │ txt + FFN   │                          │  │   │
│   │   │   │ (MLP block) │           │ (MLP block) │                          │  │   │
│   │   │   └──────┬──────┘           └──────┬──────┘                          │  │   │
│   │   │          │                         │                                 │  │   │
│   │   │          ▼                         ▼                                 │  │   │
│   │   │   img [B,N,D]               txt [B,L,D]                              │  │   │
│   │   │                                                                      │  │   │
│   │   └──────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          5. SingleStreamBlock (×38)                          │   │
│   ├─────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                             │   │
│   │   单流处理: img 和 txt concat 后一起处理                                    │   │
│   │                                                                             │   │
│   │   ┌──────────────────────────────────────────────────────────────────────┐  │   │
│   │   │                                                                      │  │   │
│   │   │   x = concat([img, txt], dim=1)   # [B, N+L, D]                      │  │   │
│   │   │       │                                                              │  │   │
│   │   │       ▼                                                              │  │   │
│   │   │   ┌─────────────┐                                                    │  │   │
│   │   │   │   AdaLN     │◄─────── vec                                        │  │   │
│   │   │   └──────┬──────┘                                                    │  │   │
│   │   │          │                                                           │  │   │
│   │   │          ▼                                                           │  │   │
│   │   │   ┌─────────────┐                                                    │  │   │
│   │   │   │ Self-Attn   │   (Q, K, V from x)                                 │  │   │
│   │   │   │   + RoPE    │                                                    │  │   │
│   │   │   └──────┬──────┘                                                    │  │   │
│   │   │          │                                                           │  │   │
│   │   │          ▼                                                           │  │   │
│   │   │   ┌─────────────┐                                                    │  │   │
│   │   │   │ x + gate×   │                                                    │  │   │
│   │   │   │ (attn+mlp)  │   (并行 attention 和 MLP)                          │  │   │
│   │   │   └──────┬──────┘                                                    │  │   │
│   │   │          │                                                           │  │   │
│   │   │          ▼                                                           │  │   │
│   │   │   img, txt = split(x)   # 分离回 img 和 txt                          │  │   │
│   │   │                                                                      │  │   │
│   │   └──────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          6. 输出投影                                         │   │
│   ├─────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                             │   │
│   │   只需要 img 输出 (txt 丢弃):                                               │   │
│   │                                                                             │   │
│   │   img [B, N, D]     AdaLN      Linear(D, 64)      v_pred [B, N, 64]         │   │
│   │   ─────────────►  ──────────►  ──────────────►   ───────────────────►       │   │
│   │                     (vec)       final_layer                                 │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 输入/输出规格

### 模型输入

| 输入 | 形状 | 描述 |
|-----|------|------|
| `img` | `[B, N, 64]` | 加噪后的视频 tokens (x_t) |
| `img_ids` | `[B, N, 3]` | 视频位置索引 (t, h, w) |
| `txt` | `[B, L, 4096]` | T5 文本嵌入 |
| `txt_ids` | `[B, L, 3]` | 文本位置索引 (全零) |
| `timesteps` | `[B]` | 时间步 |
| `y_vec` | `[B, 768]` | CLIP 文本嵌入 |
| `guidance` | `[B]` | 引导强度 |
| `cond` | `[B, N, 68]` | 视觉条件 (可选) |

### 模型输出

| 输出 | 形状 | 描述 |
|-----|------|------|
| `v_pred` | `[B, N, 64]` | 预测的 velocity |

## 关键代码

### MMDiT 前向传播

```python
# 位置: opensora/models/mmdit/model.py

class MMDiTModel(nn.Module):
    def forward(
        self,
        img: Tensor,        # [B, N, 64]
        img_ids: Tensor,    # [B, N, 3]
        txt: Tensor,        # [B, L, 4096]
        txt_ids: Tensor,    # [B, L, 3]
        timesteps: Tensor,  # [B]
        y_vec: Tensor,      # [B, 768]
        guidance: Tensor,   # [B]
        cond: Tensor = None,  # [B, N, 68]
    ) -> Tensor:
        
        # 1. 输入投影
        img = self.img_in(img)  # [B, N, D]
        if cond is not None:
            cond = self.cond_in(cond)  # [B, N, D]
            img = img + cond
        txt = self.txt_in(txt)  # [B, L, D]
        
        # 2. 时间与条件嵌入
        vec = self.time_in(timestep_embedding(timesteps, 256))  # [B, D]
        vec = vec + self.vector_in(y_vec)  # + CLIP
        if self.guidance_embed:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        
        # 3. 位置编码
        ids = torch.cat([txt_ids, img_ids], dim=1)  # [B, N+L, 3]
        pe = self.pe_embedder(ids)  # 3D RoPE
        
        # 4. DoubleStreamBlocks (19层)
        for block in self.double_blocks:
            img, txt = block(img, txt, vec, pe)
        
        # 5. Concat + SingleStreamBlocks (38层)
        x = torch.cat([txt, img], dim=1)  # [B, N+L, D]
        for block in self.single_blocks:
            x = block(x, vec, pe)
        
        # 6. 分离 + 输出投影
        img = x[:, txt.shape[1]:, :]  # 取 img 部分
        img = self.final_layer(img, vec)  # [B, N, 64]
        
        return img
```

### DoubleStreamBlock

```python
# 位置: opensora/models/mmdit/model.py

class DoubleStreamBlock(nn.Module):
    def forward(self, img, txt, vec, pe):
        # AdaLN 调制
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        
        # 预归一化
        img_norm = modulate(self.img_norm1(img), img_mod1)
        txt_norm = modulate(self.txt_norm1(txt), txt_mod1)
        
        # QKV 投影
        img_qkv = self.img_attn.qkv(img_norm)
        txt_qkv = self.txt_attn.qkv(txt_norm)
        
        # Joint Attention
        q = torch.cat([txt_qkv[:, :, :self.head_dim], 
                       img_qkv[:, :, :self.head_dim]], dim=1)
        k = torch.cat([txt_qkv[:, :, self.head_dim:2*self.head_dim],
                       img_qkv[:, :, self.head_dim:2*self.head_dim]], dim=1)
        v = torch.cat([txt_qkv[:, :, 2*self.head_dim:],
                       img_qkv[:, :, 2*self.head_dim:]], dim=1)
        
        # RoPE
        q, k = apply_rope(q, k, pe)
        
        # Attention
        attn_out = attention(q, k, v)
        txt_attn, img_attn = attn_out.split([txt.shape[1], img.shape[1]], dim=1)
        
        # 残差 + Gate
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        
        # FFN
        img = img + img_mod2.gate * self.img_mlp(
            modulate(self.img_norm2(img), img_mod2)
        )
        txt = txt + txt_mod2.gate * self.txt_mlp(
            modulate(self.txt_norm2(txt), txt_mod2)
        )
        
        return img, txt
```

### SingleStreamBlock

```python
# 位置: opensora/models/mmdit/model.py

class SingleStreamBlock(nn.Module):
    def forward(self, x, vec, pe):
        # AdaLN 调制
        mod, _ = self.modulation(vec)
        
        # 预归一化
        x_norm = modulate(self.pre_norm(x), mod)
        
        # 并行计算 Attention 和 MLP
        qkv, mlp = self.linear1(x_norm).split([3*self.hidden_dim, self.mlp_hidden_dim], dim=-1)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # RoPE
        q, k = apply_rope(q, k, pe)
        
        # Attention
        attn_out = attention(q, k, v)
        
        # 合并输出
        output = self.linear2(torch.cat([attn_out, self.mlp_act(mlp)], dim=-1))
        
        # 残差 + Gate
        x = x + mod.gate * output
        
        return x
```

### AdaLN-Zero 调制

```python
# 位置: opensora/models/mmdit/model.py

class Modulation(nn.Module):
    def __init__(self, dim, double=True):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim)
    
    def forward(self, vec):
        # vec: [B, D]
        out = self.lin(F.silu(vec))  # [B, multiplier*D]
        
        if self.is_double:
            # 返回两组 (shift, scale, gate)
            shift1, scale1, gate1, shift2, scale2, gate2 = out.chunk(6, dim=-1)
            return ModOut(shift1, scale1, gate1), ModOut(shift2, scale2, gate2)
        else:
            shift, scale, gate = out.chunk(3, dim=-1)
            return ModOut(shift, scale, gate), None

def modulate(x, mod: ModOut):
    """应用 AdaLN 调制"""
    return (1 + mod.scale) * x + mod.shift
```

## 模型架构参数

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              MMDiT 架构参数                                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │   hidden_size (D):        3072                                              │   │
│   │   num_heads:              24                                                │   │
│   │   head_dim:               128                                               │   │
│   │   mlp_ratio:              4.0                                               │   │
│   │   mlp_hidden_dim:         3072 × 4 = 12288                                  │   │
│   │                                                                             │   │
│   │   double_blocks:          19 层 (DoubleStreamBlock)                         │   │
│   │   single_blocks:          38 层 (SingleStreamBlock)                         │   │
│   │   总层数:                 57 层                                             │   │
│   │                                                                             │   │
│   │   in_channels:            64 (16 × 2 × 2, patch)                            │   │
│   │   context_in_dim:         4096 (T5)                                         │   │
│   │   vec_in_dim:             768 (CLIP)                                        │   │
│   │                                                                             │   │
│   │   axes_dim:               [16, 24, 24]  (T, H, W for RoPE)                  │   │
│   │   theta:                  10000                                             │   │
│   │                                                                             │   │
│   │   总参数量:               ~8B                                               │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Attention 计算

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              Joint Attention 计算                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   输入:                                                                             │
│   - img: [B, N, D]  (N=1280 for 256px 17frames)                                    │
│   - txt: [B, L, D]  (L=512)                                                        │
│                                                                                     │
│   1. QKV 分别计算:                                                                  │
│      img_q, img_k, img_v = self.img_attn.qkv(img).split(D, dim=-1)                 │
│      txt_q, txt_k, txt_v = self.txt_attn.qkv(txt).split(D, dim=-1)                 │
│                                                                                     │
│   2. 拼接:                                                                          │
│      q = [txt_q, img_q]  # [B, L+N, D]                                             │
│      k = [txt_k, img_k]  # [B, L+N, D]                                             │
│      v = [txt_v, img_v]  # [B, L+N, D]                                             │
│                                                                                     │
│   3. 多头重排:                                                                      │
│      q = q.view(B, L+N, num_heads, head_dim)                                       │
│      k = k.view(B, L+N, num_heads, head_dim)                                       │
│      v = v.view(B, L+N, num_heads, head_dim)                                       │
│                                                                                     │
│   4. RoPE:                                                                          │
│      q, k = apply_rope(q, k, pe)                                                   │
│                                                                                     │
│   5. Scaled Dot-Product Attention:                                                  │
│      attn = softmax(q @ k.T / sqrt(head_dim)) @ v                                  │
│                                                                                     │
│   6. 输出重排:                                                                      │
│      attn = attn.view(B, L+N, D)                                                   │
│      txt_out, img_out = attn.split([L, N], dim=1)                                  │
│                                                                                     │
│   Attention 矩阵形状: [B, num_heads, L+N, L+N]                                      │
│   对于 256px 17frames: [B, 24, 1792, 1792]                                          │
│                                                                                     │
│   计算复杂度: O((L+N)² × D) = O(1792² × 3072) ≈ 10B FLOPs per layer               │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 维度变换完整示例

以 batch size 16、17 帧 256×256 视频为例：

```
输入:
- img (x_t): [16, 1280, 64]
- txt: [16, 512, 4096]
- timesteps: [16]
- y_vec: [16, 768]

1. 输入投影
   img: [16, 1280, 64] → [16, 1280, 3072]
   txt: [16, 512, 4096] → [16, 512, 3072]
   cond: [16, 1280, 68] → [16, 1280, 3072]
   img = img + cond: [16, 1280, 3072]

2. 时间与条件嵌入
   time_emb: [16] → [16, 3072]
   clip_emb: [16, 768] → [16, 3072]
   vec = time_emb + clip_emb: [16, 3072]

3. 位置编码
   ids: [16, 1792, 3]  (512 + 1280)
   pe: [16, 1792, 1, 64]

4. DoubleStreamBlocks (×19)
   img: [16, 1280, 3072]
   txt: [16, 512, 3072]
   每层 Attention: Q,K,V concat 后 [16, 1792, 3072]

5. SingleStreamBlocks (×38)
   x = concat([txt, img]): [16, 1792, 3072]
   每层处理整个序列

6. 输出
   img = x[:, 512:]: [16, 1280, 3072]
   v_pred = final_layer(img): [16, 1280, 64]
```
