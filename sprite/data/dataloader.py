"""
DataLoader 工厂函数和工具
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, List
from pathlib import Path

from ..config import SpriteConfig, get_default_config
from ..tokenizer import SpriteTokenizer
from .sprite_dataset import SpriteDataset
from .webdataset_wrapper import WebDatasetWrapper


def collate_fn(batch: List[dict]) -> dict:
    """
    批处理函数
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    images = torch.stack([item["image"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "images": images
    }


def is_webdataset_path(path: str) -> bool:
    """检查路径是否为 WebDataset 格式"""
    if path is None:
        return False
    
    # 检查是否包含 glob 模式或 .tar 文件
    if "*" in path or "?" in path:
        return True
    
    path_obj = Path(path)
    
    # 检查是否是 .tar 文件
    if path_obj.suffix == ".tar":
        return True
    
    # 检查目录下是否有 .tar 文件
    if path_obj.is_dir():
        tar_files = list(path_obj.glob("*.tar"))
        return len(tar_files) > 0
    
    return False


def create_dataloader(
    config: SpriteConfig = None,
    tokenizer: SpriteTokenizer = None,
    image_dir: Optional[str] = None,
    use_synthetic: bool = True,
    synthetic_size: int = 10000,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    epoch_size: Optional[int] = None
) -> DataLoader:
    """
    创建数据加载器
    
    自动检测数据格式:
    - 如果 image_dir 包含 .tar 文件，使用 WebDataset
    - 否则使用普通图片文件夹
    - 如果 use_synthetic=True，使用合成数据
    
    Args:
        config: 配置对象
        tokenizer: 分词器
        image_dir: 图像目录或 WebDataset 路径
        use_synthetic: 是否使用合成数据
        synthetic_size: 合成数据集大小
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        epoch_size: 每个epoch样本数 (仅用于WebDataset)
    
    Returns:
        DataLoader 实例
    """
    if config is None:
        config = get_default_config()
    
    if tokenizer is None:
        tokenizer = SpriteTokenizer(config)
    
    # 检测数据格式
    if use_synthetic:
        # 使用合成数据
        dataset = SpriteDataset(
            config=config,
            tokenizer=tokenizer,
            use_synthetic=True,
            synthetic_size=synthetic_size
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    elif is_webdataset_path(image_dir):
        # 使用 WebDataset
        # 构建 glob 模式
        path = Path(image_dir)
        if path.is_dir():
            webdataset_path = str(path / "*.tar")
        elif "*" in image_dir or path.suffix == ".tar":
            webdataset_path = image_dir
        else:
            webdataset_path = str(path / "*.tar")
        
        dataset = WebDatasetWrapper(
            config=config,
            tokenizer=tokenizer,
            webdataset_path=webdataset_path,
            shuffle=shuffle,
            epoch_size=epoch_size
        )
        
        # WebDataset 是 IterableDataset，不支持 shuffle 参数
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    else:
        # 使用普通图片文件夹
        dataset = SpriteDataset(
            config=config,
            tokenizer=tokenizer,
            image_dir=image_dir,
            use_synthetic=False
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
