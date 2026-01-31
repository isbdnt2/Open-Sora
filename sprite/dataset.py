"""
Sprite Image LLM - 数据集
用于训练的图像数据集

支持格式:
- 图片文件夹 (.png, .jpg, .jpeg, .bmp, .gif)
- WebDataset (.tar 文件)
- 合成数据

注意: 此文件为兼容层，实际实现已迁移到 sprite/data/ 目录
"""

# 从新模块导入所有内容，保持向后兼容
from .data import (
    get_default_transform,
    SpriteDataset,
    WebDatasetWrapper,
    collate_fn,
    is_webdataset_path,
    create_dataloader,
)

__all__ = [
    "get_default_transform",
    "SpriteDataset",
    "WebDatasetWrapper",
    "collate_fn",
    "is_webdataset_path",
    "create_dataloader",
]


if __name__ == "__main__":
    # 测试数据集
    from .config import get_default_config
    from .tokenizer import SpriteTokenizer
    from pathlib import Path
    
    config = get_default_config()
    tokenizer = SpriteTokenizer(config)
    
    print("=" * 60)
    print("Testing SpriteDataset (Synthetic)")
    print("=" * 60)
    
    # 使用合成数据
    dataset = SpriteDataset(
        config=config,
        tokenizer=tokenizer,
        use_synthetic=True,
        synthetic_size=100
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 获取一个样本
    sample = dataset[0]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Image shape: {sample['image'].shape}")
    
    # 测试dataloader
    dataloader = create_dataloader(
        config=config,
        tokenizer=tokenizer,
        use_synthetic=True,
        synthetic_size=100,
        batch_size=8,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print(f"Batch images shape: {batch['images'].shape}")
    
    # 测试 WebDataset (如果存在)
    webdataset_path = "dataset/MNIST/webdataset"
    if Path(webdataset_path).exists():
        print("\n" + "=" * 60)
        print("Testing WebDataset")
        print("=" * 60)
        
        dataloader = create_dataloader(
            config=config,
            tokenizer=tokenizer,
            image_dir=webdataset_path,
            use_synthetic=False,
            batch_size=8,
            num_workers=0,
            epoch_size=100
        )
        
        batch = next(iter(dataloader))
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        print(f"Batch images shape: {batch['images'].shape}")