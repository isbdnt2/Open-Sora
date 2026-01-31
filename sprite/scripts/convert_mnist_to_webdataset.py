#!/usr/bin/env python
"""
将 MNIST 数据集转换为 WebDataset 格式

用法:
    python -m sprite.scripts.convert_mnist_to_webdataset \
        --mnist-dir dataset/MNIST \
        --output-dir dataset/MNIST/webdataset \
        --samples-per-shard 5000
"""

import argparse
import os
import io
from pathlib import Path

import webdataset as wds
from torchvision import datasets
from PIL import Image


def convert_mnist_to_webdataset(
    mnist_dir: str,
    output_dir: str,
    samples_per_shard: int = 5000,
    split: str = "train"
):
    """
    将 MNIST 转换为 WebDataset 格式
    
    Args:
        mnist_dir: MNIST 数据集根目录
        output_dir: 输出目录
        samples_per_shard: 每个 shard 的样本数
        split: "train" 或 "test"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 MNIST
    is_train = (split == "train")
    mnist = datasets.MNIST(
        root=str(Path(mnist_dir).parent),
        train=is_train,
        download=False
    )
    
    print(f"Converting MNIST {split} set: {len(mnist)} images")
    print(f"Output directory: {output_dir}")
    print(f"Samples per shard: {samples_per_shard}")
    
    # 计算 shard 数量
    num_shards = (len(mnist) + samples_per_shard - 1) // samples_per_shard
    
    # 创建 WebDataset 写入器
    pattern = str(output_dir / f"{split}-%05d.tar")
    
    with wds.ShardWriter(pattern, maxcount=samples_per_shard) as sink:
        for idx, (image, label) in enumerate(mnist):
            # 将 PIL Image 转换为 PNG 字节
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()
            
            # 写入样本
            sample = {
                "__key__": f"{idx:08d}",
                "png": img_bytes,
                "cls": str(label).encode("utf-8"),
            }
            sink.write(sample)
            
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1}/{len(mnist)} images...")
    
    print(f"\nConversion completed!")
    print(f"  Total images: {len(mnist)}")
    print(f"  Total shards: {num_shards}")
    print(f"  Output pattern: {pattern}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MNIST to WebDataset format"
    )
    parser.add_argument(
        "--mnist-dir", type=str, default="dataset/MNIST",
        help="MNIST dataset directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="dataset/MNIST/webdataset",
        help="Output directory for WebDataset"
    )
    parser.add_argument(
        "--samples-per-shard", type=int, default=5000,
        help="Number of samples per shard"
    )
    parser.add_argument(
        "--split", type=str, default="all", choices=["train", "test", "all"],
        help="Which split to convert"
    )
    
    args = parser.parse_args()
    
    if args.split in ["train", "all"]:
        convert_mnist_to_webdataset(
            mnist_dir=args.mnist_dir,
            output_dir=args.output_dir,
            samples_per_shard=args.samples_per_shard,
            split="train"
        )
    
    if args.split in ["test", "all"]:
        convert_mnist_to_webdataset(
            mnist_dir=args.mnist_dir,
            output_dir=args.output_dir,
            samples_per_shard=args.samples_per_shard,
            split="test"
        )


if __name__ == "__main__":
    main()
