# 用于获取完整模型结构信息的配置（不进行实际训练）
# 使用方式: torchrun --standalone --nproc_per_node 1 scripts/diffusion/get_model_info.py configs/diffusion/train/model_info.py

# ============================================================
# 完整 Open-Sora 2.0 模型配置 (与 image.py / stage1.py 一致)
# ============================================================

# 模型架构 - 完整 8B 参数配置
model = dict(
    type="flux",
    from_pretrained=None,
    strict_load=False,
    guidance_embed=False,
    fused_qkv=False,
    use_liger_rope=False,  # 禁用 liger rope，因为它可能需要 CUDA
    device_map="meta",     # 使用 meta device，不分配实际内存
    # 完整模型架构参数
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

# VAE 配置 (使用本地权重)
ae = dict(
    type="hunyuan_vae",
    from_pretrained="./ckpts/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=False,
)

# 文本编码器配置 (使用本地小模型以节省下载时间)
# 注意: 这不影响 Diffusion Model 的结构，因为 context_in_dim 是固定的
t5 = dict(
    type="text_embedder",
    from_pretrained="../../t5-small",  # 本地路径
    max_length=512,
    shardformer=False,
)

clip = dict(
    type="text_embedder",
    from_pretrained="../../openai-clip-base",  # 本地路径
    max_length=77,
)

# 最小化的数据集配置 (只用于初始化)
dataset = dict(
    type="DummyDataset",
    length=1,
)

bucket_config = {
    "256px": {1: (1.0, 1)},
}

# 优化器配置 (不实际使用)
optim = dict(
    cls="HybridAdam",
    lr=1e-5,
    eps=1e-15,
    weight_decay=0.0,
    adamw_mode=True,
)

# 训练配置
dtype = "bf16"
plugin = "zero2"
plugin_config = dict()
grad_checkpoint = False  # 获取信息时不需要
num_workers = 0
batch_size = 1
epochs = 1

# 输出配置
outputs = "outputs/model_info"
seed = 42

# ============================================================
# 特殊标记 - 告诉脚本只获取模型信息
# ============================================================
model_info_only = True
