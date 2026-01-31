# UNet 架构对比分析：经典 DDPM vs 业界前沿

## 📋 概述

本文档对比分析您项目中的 `ddpm.py` UNet 实现与业界主流的前沿 UNet 架构（如 Stable Diffusion、SDXL、SD3、Flux 等），帮助您理解差异和改进方向。

---

## 1. 架构总览对比

| 特性 | 您的实现 (DDPM UNet) | SD 1.x/2.x UNet | SDXL UNet | SD3/Flux (DiT) |
|------|---------------------|-----------------|-----------|----------------|
| **架构类型** | 经典 UNet | 改进 UNet | 大规模 UNet | DiT (Transformer) |
| **参数量级** | ~10M | ~860M | ~2.6B | ~2-12B |
| **注意力类型** | Self-Attention | Cross-Attention | Cross-Attention | Full Attention |
| **条件注入** | 仅 Timestep | Timestep + Text | Timestep + Text + Size | Timestep + Text |
| **归一化** | GroupNorm | GroupNorm | GroupNorm | AdaLN / RMSNorm |
| **位置编码** | 无 | 无 | 无 | RoPE |

---

## 2. 详细组件对比

### 2.1 时间步嵌入 (Timestep Embedding)

#### 您的实现
```python
# 正弦位置编码 + 2层 MLP
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(...) / half)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding

# MLP 投影
self.time_embed = nn.Sequential(
    nn.Linear(model_channels, time_embed_dim),
    nn.SiLU(),
    nn.Linear(time_embed_dim, time_embed_dim),
)
```

#### 前沿实现 (SD3/Flux 风格)
```python
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        # 支持连续时间步 (Flow Matching)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)
```

**主要区别：**
| 方面 | 您的实现 | 前沿实现 |
|------|----------|----------|
| 时间步范围 | 离散 [0, T] | 连续 [0, 1] (Flow Matching) |
| 嵌入维度 | `model_channels` | 独立的 `frequency_embedding_size` |
| 调制方式 | 加法注入 | AdaLN 调制 (scale + shift) |

---

### 2.2 残差块 (Residual Block)

#### 您的实现
```python
class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),  # GroupNorm
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        # 时间嵌入：简单的加法
        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, out_channels))
        
        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        h = self.conv1(x)
        h += self.time_emb(t)[:, :, None, None]  # 简单加法
        h = self.conv2(h)
        return h + self.shortcut(x)
```

#### 前沿实现 (SDXL 风格)
```python
class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, groups=32):
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # AdaGN: Adaptive Group Normalization
        self.time_emb_proj = nn.Linear(temb_channels, out_channels * 2)  # scale + shift
        
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
    def forward(self, x, temb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # AdaGN 调制
        temb_proj = self.time_emb_proj(F.silu(temb))
        scale, shift = temb_proj.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)
```

**主要区别：**
| 方面 | 您的实现 | 前沿实现 |
|------|----------|----------|
| 时间调制 | 加法 (`h += t_emb`) | AdaGN (scale × norm + shift) |
| 调制位置 | conv1 之后 | norm2 之后 |
| 表达能力 | 较弱 | 更强的条件控制 |

---

### 2.3 注意力机制 (Attention)

#### 您的实现
```python
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        
        # 手动实现注意力
        scale = 1.0 / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        
        return self.proj(h.reshape(B, -1, H, W)) + x
```

#### 前沿实现 (SD3/Flux 风格)
```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_norm=True):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 分离的 Q, K, V 投影
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        
        # QK Normalization (稳定训练)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x, context=None, freqs_cis=None):
        q = self.to_q(x)
        k = self.to_k(context if context is not None else x)
        v = self.to_v(context if context is not None else x)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # RoPE 位置编码
        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)
        
        # Flash Attention
        out = F.scaled_dot_product_attention(q, k, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
```

**主要区别：**
| 方面 | 您的实现 | 前沿实现 |
|------|----------|----------|
| 注意力类型 | 仅 Self-Attention | Self + Cross Attention |
| QKV 投影 | 合并的 Conv2d | 分离的 Linear |
| 位置编码 | 无 | RoPE (旋转位置编码) |
| QK 归一化 | 无 | RMSNorm (稳定大模型训练) |
| 计算优化 | 手动 einsum | Flash Attention / SDPA |
| 条件注入 | 无 | 文本 context 作为 KV |

---

### 2.4 条件注入机制

#### 您的实现
```
仅支持 Timestep 条件：
timestep → sinusoidal embedding → MLP → 加到 ResBlock
```

#### 前沿实现 (多条件)
```
1. Timestep 条件:
   t → sinusoidal → MLP → AdaLN 调制

2. 文本条件 (Cross-Attention):
   text → CLIP/T5 Encoder → Cross-Attention KV

3. 图像条件 (IP-Adapter 风格):
   image → Image Encoder → 与 KV concat

4. 额外条件 (SDXL):
   - original_size, crop_coords → 嵌入
   - micro_conditioning → pooled text embed 拼接
```

---

### 2.5 架构布局

#### 您的实现
```
channel_mult=(1, 2, 2, 2)  →  128 → 256 → 256 → 256
num_res_blocks=2
attention_resolutions=(8, 16)  →  仅在低分辨率添加注意力
```

#### 前沿实现 (SDXL)
```python
# SDXL UNet 配置
block_out_channels = (320, 640, 1280)
layers_per_block = 2
transformer_layers_per_block = [1, 2, 10]  # 更深的 Transformer
attention_head_dim = [5, 10, 20]
cross_attention_dim = 2048  # 更大的文本嵌入

# SD3/Flux (纯 Transformer)
hidden_size = 3072
num_layers = 24  # 24 个 DiT blocks
num_attention_heads = 24
```

---

## 3. 关键技术差距

### 3.1 缺失的核心功能

| 功能 | 您的实现 | 前沿实现 | 重要性 |
|------|----------|----------|--------|
| **Cross-Attention** | ❌ | ✅ | 🔴 关键 - 文本到图像生成的核心 |
| **AdaLN 调制** | ❌ (简单加法) | ✅ | 🔴 关键 - 更好的条件控制 |
| **QK Normalization** | ❌ | ✅ | 🟡 重要 - 大模型训练稳定性 |
| **RoPE 位置编码** | ❌ | ✅ | 🟡 重要 - 任意分辨率生成 |
| **Flash Attention** | ❌ | ✅ | 🟢 优化 - 显存和速度 |
| **VAE 潜空间** | ❌ (像素空间) | ✅ | 🔴 关键 - 高分辨率生成 |
| **多尺度时间嵌入** | ❌ | ✅ | 🟡 重要 - 更好的调度 |

### 3.2 架构演进趋势

```
DDPM UNet (2020)          →  您的实现
    ↓
Stable Diffusion 1.x (2022) →  添加 Cross-Attention, VAE
    ↓
Stable Diffusion 2.x (2022) →  更大的 text encoder
    ↓
SDXL (2023)               →  更大的 UNet, 双 text encoder
    ↓
SD3 / Flux (2024)         →  MM-DiT (纯 Transformer)
    ↓
Z-Image (2025)            →  优化的 DiT + Turbo 蒸馏
```

---

## 4. 改进建议

### 4.1 短期改进 (保持 UNet 架构)

```python
# 1. 添加 AdaLN 调制
class ImprovedResidualBlock(nn.Module):
    def forward(self, x, temb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # AdaLN 替代简单加法
        scale, shift = self.time_emb(F.silu(temb)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[..., None, None]) + shift[..., None, None]
        
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

# 2. 添加 Cross-Attention
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim):
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        
    def forward(self, x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        return F.scaled_dot_product_attention(q, k, v)

# 3. 使用 Flash Attention
# 替换手动 einsum 为:
out = F.scaled_dot_product_attention(q, k, v)
```

### 4.2 中期改进 (迁移到 Latent Diffusion)

```python
# 1. 添加 VAE
class AutoencoderKL:
    def encode(self, x):
        # 图像 → 潜空间 (8x 下采样)
        return self.encoder(x)
    
    def decode(self, z):
        # 潜空间 → 图像
        return self.decoder(z)

# 2. 训练流程改变
# 原始: noise_pred = model(x_noisy, t)
# 改进: noise_pred = model(z_noisy, t, text_embed)
```

### 4.3 长期演进 (DiT 架构)

参考 Z-Image 的实现：
- 纯 Transformer 架构
- Patchify + 位置编码
- 双向注意力 (图像 + 文本)
- Flow Matching 训练

---

## 5. 代码结构对比图

### 您的 UNet
```
Input (x, t)
    │
    ├─► Timestep Embedding ─► time_emb
    │
    ▼
┌─────────────────────────────────┐
│         Down Blocks             │
│  ┌─────────────────────────┐    │
│  │ ResBlock + time_emb add │    │
│  │ [Optional] Self-Attn    │    │
│  │ Downsample              │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
    │ (skip connections)
    ▼
┌─────────────────────────────────┐
│        Middle Block             │
│  ResBlock → Self-Attn → ResBlock│
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│          Up Blocks              │
│  ┌─────────────────────────┐    │
│  │ Concat skip connection  │    │
│  │ ResBlock + time_emb add │    │
│  │ [Optional] Self-Attn    │    │
│  │ Upsample                │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
    │
    ▼
Output (noise prediction)
```

### SD3/Flux DiT
```
Input (x, t, text)
    │
    ├─► Timestep Embedding ─► adaln_input
    ├─► Text Encoder ────────► text_embeds
    │
    ▼
┌─────────────────────────────────┐
│       Patchify + Embed          │
│  x: [B,C,H,W] → [B, N, D]       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│    DiT Blocks (× N layers)      │
│  ┌─────────────────────────┐    │
│  │ AdaLN(x, adaln_input)   │    │
│  │ Self-Attention + RoPE   │    │
│  │ Cross-Attention(x, text)│    │
│  │ AdaLN + FFN             │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│       Final Layer               │
│  AdaLN → Linear → Unpatchify    │
└─────────────────────────────────┘
    │
    ▼
Output (velocity / noise)
```

---

## 6. 总结

您的实现是一个**经典的 DDPM UNet**，适合学习和理解扩散模型的基本原理。与业界前沿相比，主要差距在于：

1. **条件机制**: 缺少 Cross-Attention，无法进行文本引导生成
2. **调制方式**: 使用简单加法而非 AdaLN，条件控制能力较弱
3. **规模差异**: 参数量级差距 (~10M vs ~2B+)
4. **训练范式**: 像素空间 DDPM vs 潜空间 Flow Matching
5. **优化技术**: 缺少 Flash Attention、QK Norm 等现代优化

建议根据您的目标：
- **学习目的**: 当前实现足够理解核心概念
- **实际应用**: 考虑使用 diffusers 库的预训练模型
- **深入研究**: 参考 SD3/Flux 的 DiT 架构进行改进
