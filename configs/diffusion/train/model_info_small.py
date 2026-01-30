# 用于获取模型结构信息的配置 - 使用 meta device 零内存占用
# 使用方式: python scripts/diffusion/get_model_info.py configs/diffusion/train/model_info_small.py
#
# 注意: 使用 meta device，模型参数不占用实际内存（仅需约 100MB）
# 可以获取完整的模型结构和参数统计，但无法运行前向传播

# ============================================================
# 完整 Open-Sora 2.0 模型配置 (Meta Device - 零内存)
# ============================================================

model = dict(
    type="flux",
    from_pretrained=None,
    strict_load=False,
    guidance_embed=False,
    fused_qkv=False,
    use_liger_rope=False,  # 禁用 liger rope（使用 mock）
    device_map="meta",     # 使用 meta device，零内存占用！
    # 完整模型架构参数 (与 image.py 一致)
    in_channels=64,              # latent_channels(16) * patch_size^2(4)
    vec_in_dim=768,              # CLIP 输出维度
    context_in_dim=4096,         # T5-XXL 输出维度
    hidden_size=3072,            # Transformer 隐藏层维度
    mlp_ratio=4.0,               # MLP 扩展比例
    num_heads=24,                # 注意力头数
    depth=19,                    # Double Stream Blocks 数量
    depth_single_blocks=38,      # Single Stream Blocks 数量
    axes_dim=[16, 56, 56],       # RoPE 位置编码维度 [T, H, W]
    theta=10_000,                # RoPE theta
    qkv_bias=True,
)

# 输出配置
outputs = "outputs/model_info"
use_meta_device = True  # 使用 meta device，不运行前向传播
