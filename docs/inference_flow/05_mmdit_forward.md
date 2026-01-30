# 05. MMDiT 模型前向传播 (MMDiT Forward Pass)

## 流程图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                             MMDiT 模型前向传播                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          1. 输入投影                                         │   │
│   ├─────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                             │   │
│   │   img [B, N, 64]              txt [B, seq_len, 4096]                       │   │
│   │        │                              │                                     │   │
│   │        ▼ img_in                       ▼ txt_in                              │   │
│   │   Linear(64, 3072)              Linear(4096, 3072)                         │   │
│   │        │                              │                                     │   │
│   │        ▼                              ▼                                     │   │
│   │   img [B, N, 3072]              txt [B, seq_len, 3072]                     │   │
│   │                                                                             │   │
│   │   timestep [B]                  y_vec [B, 768]                             │   │
│   │        │                              │                                     │   │
│   │        ▼ time_in                      ▼ vector_in                           │   │
│   │   MLPEmbedder(256→3072)         MLPEmbedder(768→3072)                      │   │
│   │        │                              │                                     │   │
│   │        └───────────┬──────────────────┘                                     │   │
│   │                    ▼                                                        │   │
│   │               vec [B, 3072]  (时间步 + CLIP 嵌入)                           │   │
│   │                                                                             │   │
│   │   cond [B, N, 68]  (可选，I2V 条件)                                         │   │
│   │        │                                                                    │   │
│   │        ▼ cond_in                                                            │   │
│   │   Linear(68, 3072)                                                          │   │
│   │        │                                                                    │   │
│   │        ▼                                                                    │   │
│   │   img = img + cond_proj  (条件注入)                                         │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          2. 位置编码                                         │   │
│   ├─────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                             │   │
│   │   img_ids [B, N, 3]    txt_ids [B, seq_len, 3]                              │   │
│   │        │                       │                                            │   │
│   │        └───────────┬───────────┘                                            │   │
│   │                    ▼ concat                                                 │   │
│   │            ids [B, N+seq_len, 3]                                            │   │
│   │                    │                                                        │   │
│   │                    ▼ pe_embedder (RoPE)                                     │   │
│   │            pe [B, N+seq_len, num_heads, head_dim]                           │   │
│   │                                                                             │   │
│   │   3D RoPE: 每个位置 (t, h, w) 编码到 attention 的 Q/K 中                    │   │
│   │   axes_dim = [16, 56, 56] → head_dim = 128                                  │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                    3. DoubleStreamBlock × depth (19层)                      │   │
│   ├─────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                             │   │
│   │   ┌─────────────────────────────────────────────────────────────────────┐   │   │
│   │   │                      DoubleStreamBlock                              │   │   │
│   │   │                                                                     │   │   │
│   │   │   Image Stream                           Text Stream                │   │   │
│   │   │   img [B, N, 3072]                       txt [B, seq_len, 3072]     │   │   │
│   │   │        │                                       │                    │   │   │
│   │   │        ▼ img_mod(vec)                          ▼ txt_mod(vec)       │   │   │
│   │   │   (生成 shift, scale, gate)              (生成 shift, scale, gate)  │   │   │
│   │   │        │                                       │                    │   │   │
│   │   │        ▼ LayerNorm + Modulation                ▼ LayerNorm + Mod    │   │   │
│   │   │        │                                       │                    │   │   │
│   │   │        ▼ Q, K, V                               ▼ Q, K, V            │   │   │
│   │   │        │                                       │                    │   │   │
│   │   │        └───────────────┬───────────────────────┘                    │   │   │
│   │   │                        ▼                                            │   │   │
│   │   │              ┌─────────────────────┐                                │   │   │
│   │   │              │  Joint Attention    │                                │   │   │
│   │   │              │  concat([img, txt]) │                                │   │   │
│   │   │              │  全序列 attention   │                                │   │   │
│   │   │              │  + RoPE             │                                │   │   │
│   │   │              └─────────────────────┘                                │   │   │
│   │   │                        │                                            │   │   │
│   │   │        ┌───────────────┴───────────────────────┐                    │   │   │
│   │   │        ▼ img_proj                              ▼ txt_proj           │   │   │
│   │   │        │                                       │                    │   │   │
│   │   │        ▼ img_mlp                               ▼ txt_mlp            │   │   │
│   │   │   3072→12288→3072                         3072→12288→3072           │   │   │
│   │   │        │                                       │                    │   │   │
│   │   │        ▼                                       ▼                    │   │   │
│   │   │   img (更新后)                             txt (更新后)             │   │   │
│   │   │                                                                     │   │   │
│   │   └─────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                             │   │
│   │   重复 depth=19 次                                                          │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                  4. SingleStreamBlock × depth_single (38层)                 │   │
│   ├─────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                             │   │
│   │   ┌─────────────────────────────────────────────────────────────────────┐   │   │
│   │   │                      SingleStreamBlock                              │   │   │
│   │   │                                                                     │   │   │
│   │   │   x = concat([img, txt])  [B, N+seq_len, 3072]                      │   │   │
│   │   │        │                                                            │   │   │
│   │   │        ▼ modulation(vec)                                            │   │   │
│   │   │   (生成 shift, scale, gate)                                         │   │   │
│   │   │        │                                                            │   │   │
│   │   │        ▼ LayerNorm + Mod                                            │   │   │
│   │   │        │                                                            │   │   │
│   │   │        ├───────────────────────────────────────┐                    │   │   │
│   │   │        ▼ QKV + Attention                       ▼ MLP                │   │   │
│   │   │        │                                       │                    │   │   │
│   │   │        └───────────────┬───────────────────────┘                    │   │   │
│   │   │                        ▼ concat + proj                              │   │   │
│   │   │                        │                                            │   │   │
│   │   │                        ▼                                            │   │   │
│   │   │                   x (更新后)                                        │   │   │
│   │   │                                                                     │   │   │
│   │   └─────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                             │   │
│   │   重复 depth_single_blocks=38 次                                            │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          5. 输出层                                           │   │
│   ├─────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                             │   │
│   │   x = x[:, :N]  (只取图像部分)                                              │   │
│   │        │                                                                    │   │
│   │        ▼ final_layer                                                        │   │
│   │   LayerNorm + Modulation(vec) + Linear(3072, 64)                            │   │
│   │        │                                                                    │   │
│   │        ▼                                                                    │   │
│   │   output [B, N, 64]  (预测的速度场)                                         │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 输入/输出规格

### 输入

| 参数 | 形状 | 描述 |
|-----|------|------|
| `img` | `[B, N, in_channels]` | Patchified 噪声图像 |
| `img_ids` | `[B, N, 3]` | 图像位置 (t, h, w) |
| `txt` | `[B, seq_len, context_dim]` | T5 文本嵌入 |
| `txt_ids` | `[B, seq_len, 3]` | 文本位置 (全0) |
| `y_vec` | `[B, vec_dim]` | CLIP 全局向量 |
| `timesteps` | `[B]` | 当前时间步 |
| `guidance` | `[B]` | CFG 强度 (可选) |
| `cond` | `[B, N, cond_dim]` | I2V 条件 (可选) |

### 输出

| 参数 | 形状 | 描述 |
|-----|------|------|
| `output` | `[B, N, in_channels]` | 预测的速度场 |

## 关键代码

### 模型结构

```python
# 位置: opensora/models/mmdit/model.py

class MMDiTModel(nn.Module):
    def __init__(self, config: MMDiTConfig):
        super().__init__()
        
        # 输入投影
        self.img_in = nn.Linear(config.in_channels, config.hidden_size)  # 64 → 3072
        self.txt_in = nn.Linear(config.context_in_dim, config.hidden_size)  # 4096 → 3072
        
        # 时间步和 CLIP 嵌入
        self.time_in = MLPEmbedder(256, config.hidden_size)  # 256 → 3072
        self.vector_in = MLPEmbedder(config.vec_in_dim, config.hidden_size)  # 768 → 3072
        
        # 可选: Guidance 嵌入 (用于蒸馏模型)
        self.guidance_in = (
            MLPEmbedder(256, config.hidden_size)
            if config.guidance_embed else nn.Identity()
        )
        
        # 可选: 条件嵌入 (I2V)
        self.cond_in = (
            nn.Linear(config.in_channels + config.patch_size**2, config.hidden_size)
            if config.cond_embed else nn.Identity()
        )
        
        # 位置编码
        self.pe_embedder = EmbedND(
            dim=config.hidden_size // config.num_heads,  # head_dim
            theta=config.theta,
            axes_dim=config.axes_dim
        )
        
        # Double Stream Blocks
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(config.hidden_size, config.num_heads, ...)
            for _ in range(config.depth)  # 19 层
        ])
        
        # Single Stream Blocks  
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(config.hidden_size, config.num_heads, ...)
            for _ in range(config.depth_single_blocks)  # 38 层
        ])
        
        # 输出层
        self.final_layer = LastLayer(config.hidden_size, 1, config.in_channels)
```

### 前向传播

```python
# 位置: opensora/models/mmdit/model.py

def prepare_block_inputs(self, img, img_ids, txt, txt_ids, timesteps, y_vec, cond=None, guidance=None):
    """准备 Transformer blocks 的输入"""
    
    # 1. 图像投影
    img = self.img_in(img)  # [B, N, 64] → [B, N, 3072]
    
    # 2. 条件注入 (I2V)
    if self.config.cond_embed and cond is not None:
        img = img + self.cond_in(cond)
    
    # 3. 时间步嵌入
    vec = self.time_in(timestep_embedding(timesteps, 256))  # [B, 3072]
    
    # 4. 可选: Guidance 嵌入
    if self.config.guidance_embed and guidance is not None:
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
    
    # 5. CLIP 嵌入
    vec = vec + self.vector_in(y_vec)  # [B, 3072]
    
    # 6. 文本投影
    txt = self.txt_in(txt)  # [B, seq_len, 4096] → [B, seq_len, 3072]
    
    # 7. 位置编码 (RoPE)
    ids = torch.cat((txt_ids, img_ids), dim=1)  # [B, seq_len+N, 3]
    pe = self.pe_embedder(ids)  # [B, seq_len+N, num_heads, head_dim]
    
    return img, txt, vec, pe


def forward_ckpt(self, **kwargs):
    """前向传播 (带梯度检查点)"""
    
    img, txt, vec, pe = self.prepare_block_inputs(**kwargs)
    
    # Double Stream Blocks
    for block in self.double_blocks:
        img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
    
    # 合并图像和文本
    img = torch.cat((txt, img), dim=1)
    
    # Single Stream Blocks
    for block in self.single_blocks:
        img = block(img, vec=vec, pe=pe)
    
    # 只取图像部分
    img = img[:, txt.shape[1]:, ...]
    
    # 输出层
    img = self.final_layer(img, vec)
    
    return img
```

## DoubleStreamBlock 详解

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           DoubleStreamBlock 详解                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌───────────────────────────────┐   ┌───────────────────────────────┐             │
│   │        Image Stream           │   │        Text Stream            │             │
│   │        img [B, N, H]          │   │        txt [B, S, H]          │             │
│   └───────────────┬───────────────┘   └───────────────┬───────────────┘             │
│                   │                                   │                             │
│                   ▼                                   ▼                             │
│   ┌───────────────────────────────┐   ┌───────────────────────────────┐             │
│   │   img_mod = modulation(vec)   │   │   txt_mod = modulation(vec)   │             │
│   │   输出 6 个参数:              │   │   输出 6 个参数:              │             │
│   │   shift1, scale1, gate1       │   │   shift1, scale1, gate1       │             │
│   │   shift2, scale2, gate2       │   │   shift2, scale2, gate2       │             │
│   └───────────────┬───────────────┘   └───────────────┬───────────────┘             │
│                   │                                   │                             │
│                   ▼                                   ▼                             │
│   ┌───────────────────────────────┐   ┌───────────────────────────────┐             │
│   │   x = LayerNorm(img)          │   │   x = LayerNorm(txt)          │             │
│   │   x = x * (1 + scale1)        │   │   x = x * (1 + scale1)        │             │
│   │       + shift1                │   │       + shift1                │             │
│   └───────────────┬───────────────┘   └───────────────┬───────────────┘             │
│                   │                                   │                             │
│                   ▼                                   ▼                             │
│   ┌───────────────────────────────┐   ┌───────────────────────────────┐             │
│   │   img_Q, img_K, img_V         │   │   txt_Q, txt_K, txt_V         │             │
│   │   = img_attn.qkv(x)           │   │   = txt_attn.qkv(x)           │             │
│   └───────────────┬───────────────┘   └───────────────┬───────────────┘             │
│                   │                                   │                             │
│                   └───────────────┬───────────────────┘                             │
│                                   ▼                                                 │
│                   ┌───────────────────────────────────┐                             │
│                   │        Joint Attention            │                             │
│                   │                                   │                             │
│                   │   Q = concat([img_Q, txt_Q])      │                             │
│                   │   K = concat([img_K, txt_K])      │                             │
│                   │   V = concat([img_V, txt_V])      │                             │
│                   │                                   │                             │
│                   │   Q, K = apply_rope(Q, K, pe)     │                             │
│                   │                                   │                             │
│                   │   attn = softmax(Q @ K.T / √d)    │                             │
│                   │   out = attn @ V                  │                             │
│                   │                                   │                             │
│                   │   [B, N+S, num_heads, head_dim]   │                             │
│                   └───────────────┬───────────────────┘                             │
│                                   │                                                 │
│                   ┌───────────────┴───────────────────┐                             │
│                   │                                   │                             │
│                   ▼                                   ▼                             │
│   ┌───────────────────────────────┐   ┌───────────────────────────────┐             │
│   │   img_attn_out = out[:, :N]   │   │   txt_attn_out = out[:, N:]   │             │
│   │   img_attn_out = proj(...)    │   │   txt_attn_out = proj(...)    │             │
│   │   img = img + gate1 * ...     │   │   txt = txt + gate1 * ...     │             │
│   └───────────────┬───────────────┘   └───────────────┬───────────────┘             │
│                   │                                   │                             │
│                   ▼                                   ▼                             │
│   ┌───────────────────────────────┐   ┌───────────────────────────────┐             │
│   │   x = LayerNorm(img)          │   │   x = LayerNorm(txt)          │             │
│   │   x = x * (1 + scale2)        │   │   x = x * (1 + scale2)        │             │
│   │       + shift2                │   │       + shift2                │             │
│   │   x = MLP(x)                  │   │   x = MLP(x)                  │             │
│   │   img = img + gate2 * x       │   │   txt = txt + gate2 * x       │             │
│   └───────────────┬───────────────┘   └───────────────┬───────────────┘             │
│                   │                                   │                             │
│                   ▼                                   ▼                             │
│   ┌───────────────────────────────┐   ┌───────────────────────────────┐             │
│   │   img (更新后) [B, N, H]      │   │   txt (更新后) [B, S, H]      │             │
│   └───────────────────────────────┘   └───────────────────────────────┘             │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Modulation 机制

```python
# 位置: opensora/models/mmdit/layers.py

class Modulation(nn.Module):
    """AdaLN-Zero 调制"""
    
    def __init__(self, hidden_size, num_modulations=6):
        super().__init__()
        # vec → 6 个调制参数
        self.silu = nn.SiLU()
        self.lin = nn.Linear(hidden_size, num_modulations * hidden_size)
    
    def forward(self, vec):
        # vec: [B, hidden_size]
        out = self.lin(self.silu(vec))  # [B, 6 * hidden_size]
        return out.chunk(6, dim=-1)  # 6 个 [B, hidden_size]

# 使用方式:
# shift1, scale1, gate1, shift2, scale2, gate2 = modulation(vec)
# x = layer_norm(x) * (1 + scale) + shift  # 调制 LayerNorm
# x = x * gate  # 门控
```

## 模型规模对比

| 参数 | dummy_test (测试) | 正式模型 | 说明 |
|------|------------------|----------|------|
| `hidden_size` | 768 | 3072 | 隐藏层维度 |
| `num_heads` | 12 | 24 | 注意力头数 |
| `depth` | 4 | 19 | DoubleStreamBlock 层数 |
| `depth_single_blocks` | 8 | 38 | SingleStreamBlock 层数 |
| `mlp_ratio` | 4.0 | 4.0 | MLP 扩展比例 |
| `axes_dim` | [16, 24, 24] | [16, 56, 56] | RoPE 各轴维度 |
| 总参数量 | ~100M | ~2B | 估算 |

## 梯度检查点设置

```python
# grad_ckpt_settings = (start_layer, end_layer)
# 例: (8, 100) 表示从第 8 层开始到第 100 层都使用梯度检查点

# 效果:
# - 节省显存: 不存储中间激活，反向传播时重新计算
# - 代价: 计算时间增加约 20-30%

# 在前向传播中:
def forward_selective_ckpt(self, **kwargs):
    img, txt, vec, pe = self.prepare_block_inputs(**kwargs)
    
    start, end = self.config.grad_ckpt_settings
    
    for i, block in enumerate(self.double_blocks):
        if start <= i < end:
            # 使用梯度检查点
            img, txt = checkpoint(block, img, txt, vec, pe)
        else:
            # 正常前向
            img, txt = block(img, txt, vec, pe)
    
    # ... SingleStreamBlocks 同理
```
