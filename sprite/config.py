"""
Sprite Image LLM - 配置文件
基于自回归Transformer的图像生成模型
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageConfig:
    """图像相关配置"""
    image_size: int = 32          # 图像尺寸 (32x32)
    patch_size: int = 2           # Patch尺寸 (2x2)
    channels: int = 1             # 通道数 (灰度图)
    
    @property
    def num_patches_per_side(self) -> int:
        """每边的patch数量"""
        return self.image_size // self.patch_size
    
    @property
    def num_patches(self) -> int:
        """总patch数量"""
        return self.num_patches_per_side ** 2
    
    @property
    def pixels_per_patch(self) -> int:
        """每个patch的像素数"""
        return self.patch_size ** 2


@dataclass
class QuantConfig:
    """量化相关配置"""
    value_step: float = 0.25      # 离散值步长
    
    @property
    def num_values(self) -> int:
        """每个像素的可能取值数"""
        return int(1.0 / self.value_step) + 1  # {0.00, 0.25, 0.50, 0.75, 1.00} = 5


@dataclass
class ModelConfig:
    """模型相关配置"""
    d_model: int = 256            # 隐藏层维度
    n_heads: int = 8              # 注意力头数
    n_layers: int = 6             # Transformer层数
    d_ff: int = 1024              # FFN中间层维度
    dropout: float = 0.1          # Dropout率
    
    # 特殊token
    bos_token_id: int = 0         # BOS token ID (会在初始化时设置)
    eos_token_id: int = 1         # EOS token ID (会在初始化时设置)
    
    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.d_model // self.n_heads


@dataclass
class TrainConfig:
    """训练相关配置"""
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # 数据
    num_workers: int = 4
    
    # 日志
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500
    
    # 路径
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"


@dataclass
class GenerateConfig:
    """生成相关配置"""
    temperature: float = 1.0
    top_k: int = 0                # 0表示不使用top-k
    top_p: float = 1.0            # 1.0表示不使用nucleus sampling


@dataclass
class SpriteConfig:
    """总配置"""
    image: ImageConfig = None
    quant: QuantConfig = None
    model: ModelConfig = None
    train: TrainConfig = None
    generate: GenerateConfig = None
    
    def __post_init__(self):
        if self.image is None:
            self.image = ImageConfig()
        if self.quant is None:
            self.quant = QuantConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.train is None:
            self.train = TrainConfig()
        if self.generate is None:
            self.generate = GenerateConfig()
        
        # 计算词表大小和特殊token
        self._setup_vocab()
    
    def _setup_vocab(self):
        """设置词表大小和特殊token ID"""
        # 词表大小 = num_values^pixels_per_patch
        num_patch_tokens = self.quant.num_values ** self.image.pixels_per_patch
        
        # 特殊token: BOS和EOS放在词表末尾
        self.model.bos_token_id = num_patch_tokens
        self.model.eos_token_id = num_patch_tokens + 1
        
        # 总词表大小
        self.vocab_size = num_patch_tokens + 2  # +2 for BOS and EOS
        
        # 最大序列长度: BOS + patches + EOS
        self.max_seq_len = 1 + self.image.num_patches + 1
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "SpriteConfig":
        """从字典创建配置"""
        image_cfg = ImageConfig(**config_dict.get("image", {}))
        quant_cfg = QuantConfig(**config_dict.get("quant", {}))
        model_cfg = ModelConfig(**config_dict.get("model", {}))
        train_cfg = TrainConfig(**config_dict.get("train", {}))
        generate_cfg = GenerateConfig(**config_dict.get("generate", {}))
        
        return cls(
            image=image_cfg,
            quant=quant_cfg,
            model=model_cfg,
            train=train_cfg,
            generate=generate_cfg
        )


def get_default_config() -> SpriteConfig:
    """获取默认配置"""
    return SpriteConfig()


if __name__ == "__main__":
    # 测试配置
    config = get_default_config()
    print(f"Image size: {config.image.image_size}x{config.image.image_size}")
    print(f"Patch size: {config.image.patch_size}x{config.image.patch_size}")
    print(f"Num patches: {config.image.num_patches}")
    print(f"Num values per pixel: {config.quant.num_values}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Max seq len: {config.max_seq_len}")
    print(f"BOS token ID: {config.model.bos_token_id}")
    print(f"EOS token ID: {config.model.eos_token_id}")
