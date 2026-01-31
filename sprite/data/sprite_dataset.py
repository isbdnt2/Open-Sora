"""
Sprite图像数据集 (Map-style)
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Optional, List
from pathlib import Path

from ..config import SpriteConfig, get_default_config
from ..tokenizer import SpriteTokenizer
from .transforms import get_default_transform


class SpriteDataset(Dataset):
    """
    Sprite图像数据集 (Map-style)
    
    支持从文件夹加载图像，或使用合成数据
    """
    
    def __init__(
        self,
        config: SpriteConfig = None,
        tokenizer: SpriteTokenizer = None,
        image_dir: Optional[str] = None,
        use_synthetic: bool = False,
        synthetic_size: int = 10000,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            config: 配置对象
            tokenizer: 分词器
            image_dir: 图像文件夹路径
            use_synthetic: 是否使用合成数据
            synthetic_size: 合成数据集大小
            transform: 图像变换
        """
        if config is None:
            config = get_default_config()
        
        if tokenizer is None:
            tokenizer = SpriteTokenizer(config)
        
        self.config = config
        self.tokenizer = tokenizer
        self.image_size = config.image.image_size
        
        # 默认变换
        if transform is None:
            self.transform = get_default_transform(self.image_size)
        else:
            self.transform = transform
        
        # 加载图像或使用合成数据
        self.use_synthetic = use_synthetic
        self.synthetic_size = synthetic_size
        
        if use_synthetic:
            self.image_paths = None
            print(f"Using synthetic dataset with {synthetic_size} samples")
        else:
            if image_dir is None:
                raise ValueError("image_dir must be provided when use_synthetic=False")
            
            self.image_dir = Path(image_dir)
            self.image_paths = self._load_image_paths()
            print(f"Loaded {len(self.image_paths)} images from {image_dir}")
    
    def _load_image_paths(self) -> List[Path]:
        """加载所有图像路径"""
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        paths = []
        
        for ext in extensions:
            paths.extend(self.image_dir.glob(f"**/*{ext}"))
            paths.extend(self.image_dir.glob(f"**/*{ext.upper()}"))
        
        return sorted(paths)
    
    def __len__(self) -> int:
        if self.use_synthetic:
            return self.synthetic_size
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """
        返回一个训练样本
        
        Returns:
            dict with:
                - input_ids: (seq_len,) token序列
                - labels: (seq_len,) 标签 (与input_ids相同)
                - image: (1, H, W) 原始图像 (用于可视化)
        """
        if self.use_synthetic:
            # 生成合成图像
            image = self._generate_synthetic_image()
        else:
            # 加载真实图像
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('L')  # 转为灰度
            image = self.transform(image)
            
            # 确保形状正确
            if image.dim() == 2:
                image = image.unsqueeze(0)
        
        # 编码为token
        tokens = self.tokenizer.encode(image)
        tokens = tokens.squeeze(0)  # 移除batch维度
        
        # input_ids: [BOS, patch1, ..., patchN, EOS] 去掉最后一个
        # labels: [patch1, ..., patchN, EOS] 去掉第一个 (BOS)
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:].clone(),
            "image": image
        }
    
    def _generate_synthetic_image(self) -> torch.Tensor:
        """
        生成合成图像用于测试
        
        生成各种简单的模式：
        - 随机噪声
        - 渐变
        - 简单形状
        """
        pattern_type = np.random.randint(0, 5)
        
        if pattern_type == 0:
            # 随机噪声
            image = torch.rand(1, self.image_size, self.image_size)
        
        elif pattern_type == 1:
            # 水平渐变
            grad = torch.linspace(0, 1, self.image_size)
            image = grad.unsqueeze(0).unsqueeze(0).expand(1, self.image_size, -1)
        
        elif pattern_type == 2:
            # 垂直渐变
            grad = torch.linspace(0, 1, self.image_size)
            image = grad.unsqueeze(0).unsqueeze(-1).expand(1, -1, self.image_size)
        
        elif pattern_type == 3:
            # 圆形
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, self.image_size),
                torch.linspace(-1, 1, self.image_size),
                indexing='ij'
            )
            radius = np.random.uniform(0.3, 0.8)
            circle = (x**2 + y**2 < radius**2).float()
            image = circle.unsqueeze(0)
        
        else:
            # 矩形
            image = torch.zeros(1, self.image_size, self.image_size)
            x1 = np.random.randint(0, self.image_size // 2)
            y1 = np.random.randint(0, self.image_size // 2)
            x2 = np.random.randint(self.image_size // 2, self.image_size)
            y2 = np.random.randint(self.image_size // 2, self.image_size)
            image[:, y1:y2, x1:x2] = np.random.uniform(0.5, 1.0)
        
        # 量化到离散值
        step = self.config.quant.value_step
        image = (torch.round(image / step) * step).clamp(0, 1)
        
        return image
