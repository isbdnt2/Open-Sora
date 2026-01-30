"""
用于测试训练流程的简化配置文件。
使用假数据集 (DummyDataset)，不需要真实视频数据。

使用方法:
    torchrun --nproc_per_node 1 --standalone scripts/diffusion/train.py configs/diffusion/train/dummy_test.py
"""

# ============================================
# Dataset settings - 使用假数据集
# ============================================
dataset = dict(
    type="dummy",
    num_samples=100,  # 假数据集大小
    num_frames=17,
    height=256,
    width=256,
    channels=3,
)

# 简化的 bucket 配置，只使用最小分辨率
bucket_config = {
    "256px": {
        1: (1.0, 2),   # (概率, batch_size) - 图片
        17: (1.0, 1),  # 短视频
    },
}

# ============================================
# Model settings - 使用更小的模型配置
# ============================================
grad_ckpt_settings = (2, 10)  # 减少 grad checkpoint 层数

model = dict(
    type="flux",
    from_pretrained=None,
    strict_load=False,
    guidance_embed=False,
    fused_qkv=False,
    use_liger_rope=False,  # 关闭 liger，避免依赖问题
    grad_ckpt_settings=grad_ckpt_settings,
    # 更小的模型架构用于测试
    # hidden_size / num_heads = head_dim，axes_dim 总和必须等于 head_dim
    # in_channels = latent_channels(16) * patch_size^2(4) = 64
    in_channels=64,
    vec_in_dim=512,        # CLIP-base 输出维度是 512
    context_in_dim=512,    # T5-small 输出维度是 512
    hidden_size=768,       # 原来是 3072，减小以节省显存
    mlp_ratio=4.0,
    num_heads=12,          # 768 / 12 = 64 (head_dim)
    depth=4,               # 原来是 19，大幅减少
    depth_single_blocks=8, # 原来是 38，大幅减少
    axes_dim=[16, 24, 24], # 16 + 24 + 24 = 64 = head_dim
    theta=10_000,
    qkv_bias=True,
)

# Dropout ratio
dropout_ratio = {
    "t5": 0.3,
    "clip": 0.3,
}

# ============================================
# VAE settings
# ============================================
ae = dict(
    type="hunyuan_vae",
    from_pretrained="./ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=False,
)
is_causal_vae = True

# ============================================
# Text encoder settings
# ============================================
t5 = dict(
    type="text_embedder",
    from_pretrained="./ckpts/t5-small",
    max_length=512,
    shardformer=False,  # 单卡测试关闭
)
clip = dict(
    type="text_embedder",
    from_pretrained="./ckpts/openai-clip-base",
    max_length=77,
)

# ============================================
# Optimization settings
# ============================================
lr = 1e-4
eps = 1e-8
optim = dict(
    cls="HybridAdam",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
    adamw_mode=True,
)
warmup_steps = 10
grad_clip = 1.0
accumulation_steps = 1
ema_decay = None  # 测试时不用 EMA

# ============================================
# Acceleration settings
# ============================================
prefetch_factor = None  # num_workers=0 时必须为 None
num_workers = 0  # 测试时用 0，避免多进程问题
num_bucket_build_workers = 1
dtype = "bf16"
plugin = "zero2"
grad_checkpoint = True
plugin_config = dict(
    reduce_bucket_size_in_m=64,
    overlap_allgather=False,
)

# ============================================
# Other settings
# ============================================
seed = 42
outputs = "outputs/dummy_test"
epochs = 2
log_every = 1
ckpt_every = 100     # 测试时不保存 checkpoint
keep_n_latest = 1
wandb = False      # 关闭 wandb
