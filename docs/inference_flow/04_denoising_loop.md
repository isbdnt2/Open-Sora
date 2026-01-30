# 04. 去噪循环 (Denoising Loop)

## 流程图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                  去噪循环流程                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   初始状态                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │   img = 纯噪声 z                                                            │   │
│   │   timesteps = [1.0, 0.98, ..., 0.02, 0.0]  (51个时间点，50步)               │   │
│   │   guidance = 4.0  (CFG 强度)                                                │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          去噪循环 (50 步)                                    │   │
│   ├─────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                             │   │
│   │   for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):│   │
│   │                                                                             │   │
│   │   ┌─────────────────────────────────────────────────────────────────────┐   │   │
│   │   │  步骤 i: t_curr → t_prev                                            │   │   │
│   │   │  例: t_curr=1.0 → t_prev=0.98 (第一步)                              │   │   │
│   │   │      t_curr=0.02 → t_prev=0.0 (最后一步)                            │   │   │
│   │   └─────────────────────────────────────────────────────────────────────┘   │   │
│   │                                │                                            │   │
│   │                                ▼                                            │   │
│   │   ┌─────────────────────────────────────────────────────────────────────┐   │   │
│   │   │  1. 准备 CFG 输入 (I2V 模式)                                        │   │   │
│   │   │                                                                     │   │   │
│   │   │  img_input = [cond_x, cond_x, cond_x]  (3倍 batch)                  │   │   │
│   │   │              ├─ 正向条件                                            │   │   │
│   │   │              ├─ 文本无条件 (仍有图像条件)                           │   │   │
│   │   │              └─ 完全无条件                                          │   │   │
│   │   │                                                                     │   │   │
│   │   │  cond_input = [cond, cond, zeros]                                   │   │   │
│   │   │              ├─ 有图像条件                                          │   │   │
│   │   │              ├─ 有图像条件                                          │   │   │
│   │   │              └─ 无图像条件                                          │   │   │
│   │   └─────────────────────────────────────────────────────────────────────┘   │   │
│   │                                │                                            │   │
│   │                                ▼                                            │   │
│   │   ┌─────────────────────────────────────────────────────────────────────┐   │   │
│   │   │  2. 模型前向传播                                                    │   │   │
│   │   │                                                                     │   │   │
│   │   │  pred = model(                                                      │   │   │
│   │   │      img=img_input,           # [3B, N, C]                          │   │   │
│   │   │      txt=txt,                 # [3B, seq_len, C]                    │   │   │
│   │   │      img_ids=img_ids,         # [3B, N, 3]                          │   │   │
│   │   │      txt_ids=txt_ids,         # [3B, seq_len, 3]                    │   │   │
│   │   │      y_vec=y_vec,             # [3B, C]                             │   │   │
│   │   │      timesteps=t_vec,         # [3B]                                │   │   │
│   │   │      guidance=guidance_vec,   # [3B]                                │   │   │
│   │   │      cond=cond_input,         # [3B, N, cond_dim]                   │   │   │
│   │   │  )                                                                  │   │   │
│   │   │  # 输出: pred [3B, N, C] - 预测的速度场                             │   │   │
│   │   └─────────────────────────────────────────────────────────────────────┘   │   │
│   │                                │                                            │   │
│   │                                ▼                                            │   │
│   │   ┌─────────────────────────────────────────────────────────────────────┐   │   │
│   │   │  3. CFG 合并                                                        │   │   │
│   │   │                                                                     │   │   │
│   │   │  cond, uncond, uncond_2 = pred.chunk(3, dim=0)                      │   │   │
│   │   │                                                                     │   │   │
│   │   │  pred = uncond_2                                                    │   │   │
│   │   │       + image_gs * (uncond - uncond_2)   # 图像引导                 │   │   │
│   │   │       + text_gs  * (cond - uncond)       # 文本引导                 │   │   │
│   │   │                                                                     │   │   │
│   │   │  默认: image_gs = guidance_img (如 4.0)                             │   │   │
│   │   │        text_gs = guidance (如 4.0)                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────────┘   │   │
│   │                                │                                            │   │
│   │                                ▼                                            │   │
│   │   ┌─────────────────────────────────────────────────────────────────────┐   │   │
│   │   │  4. Euler 更新                                                      │   │   │
│   │   │                                                                     │   │   │
│   │   │  img = img + (t_prev - t_curr) * pred                               │   │   │
│   │   │                                                                     │   │   │
│   │   │  解释:                                                              │   │   │
│   │   │  - pred 是模型预测的速度场 v                                        │   │   │
│   │   │  - (t_prev - t_curr) 是时间步长 dt (负数，因为 t 递减)              │   │   │
│   │   │  - 这是 Flow Matching 的 Euler 积分                                 │   │   │
│   │   │                                                                     │   │   │
│   │   │  数学: dx/dt = v(x, t)                                              │   │   │
│   │   │        x(t+dt) = x(t) + v * dt                                      │   │   │
│   │   └─────────────────────────────────────────────────────────────────────┘   │   │
│   │                                │                                            │   │
│   │                                ▼                                            │   │
│   │   ┌─────────────────────────────────────────────────────────────────────┐   │   │
│   │   │  重复 50 次...                                                      │   │   │
│   │   └─────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   最终状态                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │   img = 去噪后的潜在表示 (接近干净数据)                                     │   │
│   │   形状: [B, N, C]                                                           │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 去噪过程可视化

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              去噪过程可视化                                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   t=1.0        t=0.8        t=0.6        t=0.4        t=0.2        t=0.0           │
│   (纯噪声)                                                           (干净数据)     │
│                                                                                     │
│   ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐             │
│   │░░░░░│     │░▒░░░│     │▒▓▒░░│     │▓▓▒░░│     │█▓▓▒░│     │███▓░│             │
│   │░░░░░│     │░░▒░░│     │░▒▓▒░│     │▒▓▓▒░│     │▓██▓▒│     │████▓│             │
│   │░░░░░│ ──► │░░░░░│ ──► │░░▒░░│ ──► │░▒▓▒░│ ──► │▒▓█▓▒│ ──► │▓███▓│             │
│   │░░░░░│     │░░░░░│     │░░░▒░│     │░░▒▓▒│     │░▒▓█▓│     │░▓███│             │
│   │░░░░░│     │░░░░░│     │░░░░▒│     │░░░▒▓│     │░░▒▓█│     │░░▓██│             │
│   └─────┘     └─────┘     └─────┘     └─────┘     └─────┘     └─────┘             │
│                                                                                     │
│   step=0      step=10     step=20     step=30     step=40     step=50             │
│                                                                                     │
│   ░ = 高噪声区域                                                                    │
│   ▒ = 中等噪声                                                                      │
│   ▓ = 低噪声                                                                        │
│   █ = 清晰内容                                                                      │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 输入/输出规格

### 循环输入

| 参数 | 形状 | 描述 |
|-----|------|------|
| `img` | `[B, N, C]` | 当前噪声状态 |
| `timesteps` | `list[float]` | 时间步列表 |
| `guidance` | `float` | 文本 CFG 强度 |
| `guidance_img` | `float` | 图像 CFG 强度（I2V） |

### 循环输出

| 参数 | 形状 | 描述 |
|-----|------|------|
| `img` | `[B, N, C]` | 去噪后的潜在表示 |

## 关键代码

### I2V Denoiser

```python
# 位置: opensora/utils/sampling.py - I2VDenoiser.denoise()

def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
    img = kwargs.pop("img")
    timesteps = kwargs.pop("timesteps")
    guidance = kwargs.pop("guidance")
    guidance_img = kwargs.pop("guidance_img")
    
    # I2V 条件
    masks = kwargs.pop("masks")
    masked_ref = kwargs.pop("masked_ref")
    
    # 可选: 振荡 CFG
    text_osci = kwargs.pop("text_osci", False)
    image_osci = kwargs.pop("image_osci", False)
    
    patch_size = kwargs.pop("patch_size", 2)
    
    # 准备 guidance 向量
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )
    
    # 去噪循环
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        # 准备时间步向量
        t_vec = torch.full(
            (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
        )
        
        # 准备条件 (mask + reference)
        b, c, t, w, h = masked_ref.size()
        cond = torch.cat((masks, masked_ref), dim=1)  # [B, 1+16, T, H, W]
        cond = pack(cond, patch_size=patch_size)      # [B, N, (1+16)*4]
        
        # 三路 CFG: [有条件, 文本无条件, 完全无条件]
        kwargs["cond"] = torch.cat([cond, cond, torch.zeros_like(cond)], dim=0)
        
        # 复制输入用于 CFG
        cond_x = img[: len(img) // 3]
        img = torch.cat([cond_x, cond_x, cond_x], dim=0)
        
        # 模型前向传播
        pred = model(
            img=img,
            **kwargs,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        
        # CFG 合并
        text_gs = get_oscillation_gs(guidance, i) if text_osci else guidance
        image_gs = get_oscillation_gs(guidance_img, i) if image_osci else guidance_img
        
        cond, uncond, uncond_2 = pred.chunk(3, dim=0)
        pred = uncond_2 + image_gs * (uncond - uncond_2) + text_gs * (cond - uncond)
        pred = torch.cat([pred, pred, pred], dim=0)
        
        # Euler 更新
        img = img + (t_prev - t_curr) * pred
    
    # 只返回一份
    img = img[: len(img) // 3]
    return img
```

### Distilled Denoiser (简化版，无 CFG)

```python
# 位置: opensora/utils/sampling.py - DistilledDenoiser.denoise()

def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
    img = kwargs.pop("img")
    timesteps = kwargs.pop("timesteps")
    guidance = kwargs.pop("guidance")
    
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )
    
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full(
            (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
        )
        
        # 单次前向传播（模型已蒸馏，不需要 CFG）
        pred = model(
            img=img,
            **kwargs,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        
        # Euler 更新
        img = img + (t_prev - t_curr) * pred
    
    return img
```

## Classifier-Free Guidance (CFG) 详解

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           双 CFG (文本 + 图像)                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   I2V 模式使用三路 CFG:                                                             │
│                                                                                     │
│   1. cond (有条件):                                                                 │
│      - 正向文本 ✓                                                                  │
│      - 图像条件 ✓                                                                  │
│                                                                                     │
│   2. uncond (文本无条件):                                                           │
│      - 负向文本 / 空文本                                                           │
│      - 图像条件 ✓                                                                  │
│                                                                                     │
│   3. uncond_2 (完全无条件):                                                         │
│      - 负向文本 / 空文本                                                           │
│      - 无图像条件                                                                  │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                           CFG 公式                                          │   │
│   │                                                                             │   │
│   │   pred = uncond_2                           # 基准 (完全无条件)             │   │
│   │        + image_gs * (uncond - uncond_2)     # 图像引导方向                  │   │
│   │        + text_gs  * (cond - uncond)         # 文本引导方向                  │   │
│   │                                                                             │   │
│   │   等价于:                                                                   │   │
│   │   pred = (1 - image_gs - text_gs) * uncond_2                               │   │
│   │        + (image_gs) * uncond                                               │   │
│   │        + (text_gs) * cond                                                  │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   参数影响:                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │   text_gs (文本引导强度):                                                   │   │
│   │   - 值越大，生成内容越符合文本描述                                          │   │
│   │   - 太大会导致过度饱和、不自然                                              │   │
│   │   - 常用值: 4.0 ~ 7.5                                                       │   │
│   │                                                                             │   │
│   │   image_gs (图像引导强度):                                                  │   │
│   │   - 值越大，生成内容越接近参考图像                                          │   │
│   │   - 太大会限制创意自由度                                                    │   │
│   │   - 常用值: 2.0 ~ 4.0                                                       │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 振荡 CFG (Oscillation CFG)

```python
def get_oscillation_gs(guidance_scale: float, i: int, force_num=10):
    """
    前 force_num 步使用完整 guidance
    之后奇偶步交替使用 guidance 和 1.0
    
    目的: 减少过度引导导致的伪影
    """
    if i < force_num or (i >= force_num and i % 2 == 0):
        gs = guidance_scale
    else:
        gs = 1.0
    return gs

# 示例 (guidance=4.0, force_num=10):
# step 0-9:   gs = 4.0
# step 10:    gs = 4.0 (偶数)
# step 11:    gs = 1.0 (奇数)
# step 12:    gs = 4.0 (偶数)
# ...
```

## Flow Matching vs DDPM

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         Flow Matching vs DDPM 对比                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   DDPM (去噪扩散):                         Flow Matching (流匹配):                  │
│   ───────────────                         ──────────────────────                    │
│                                                                                     │
│   训练目标: 预测噪声 ε                     训练目标: 预测速度 v                     │
│   x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε        x_t = (1-t) * x_0 + t * ε               │
│                                           v = ε - x_0                              │
│                                                                                     │
│   采样: 复杂的噪声调度                     采样: 简单的线性插值                     │
│   需要预定义 β schedule                   直接 ODE 积分                            │
│                                                                                     │
│   更新规则:                               更新规则:                                │
│   x_{t-1} = f(x_t, ε_pred, t)            x_{t-dt} = x_t + v * dt                  │
│   (复杂，涉及方差)                        (简单 Euler 积分)                        │
│                                                                                     │
│   本项目使用 Flow Matching:                                                        │
│   img = img + (t_prev - t_curr) * pred                                             │
│        └── x_t ──┘  └── dt (负) ──┘  └── v ──┘                                     │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```
