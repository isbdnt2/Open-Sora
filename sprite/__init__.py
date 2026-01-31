"""
Sprite Image LLM
使用类LLM的方式生成32x32灰度图像

模块结构:
- config: 配置管理
- tokenizer: 图像到token的转换
- model: Transformer模型 (带2D RoPE)
- dataset: 数据集和数据加载
- trainer: 训练逻辑
"""

from .config import (
    SpriteConfig,
    ImageConfig,
    QuantConfig,
    ModelConfig,
    TrainConfig,
    get_default_config
)
from .tokenizer import SpriteTokenizer
from .model import SpriteModel
from .dataset import SpriteDataset, create_dataloader
from .trainer import Trainer, train


__version__ = "0.1.0"
__all__ = [
    "SpriteConfig",
    "ImageConfig",
    "QuantConfig",
    "ModelConfig",
    "TrainConfig",
    "get_default_config",
    "SpriteTokenizer",
    "SpriteModel",
    "SpriteDataset",
    "create_dataloader",
    "Trainer",
    "train"
]
