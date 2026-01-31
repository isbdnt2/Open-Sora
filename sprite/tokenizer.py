"""
Sprite Image LLM - Tokenizer
负责图像和token之间的转换
"""

import torch
import numpy as np
from typing import Union, List, Tuple

from .config import SpriteConfig, get_default_config


class SpriteTokenizer:
    """
    图像Token化器
    
    将图像分割成patch，并将每个patch量化为一个token ID
    词表大小 = num_values^pixels_per_patch + 2 (BOS, EOS)
    """
    
    def __init__(self, config: SpriteConfig = None):
        if config is None:
            config = get_default_config()
        
        self.config = config
        self.image_size = config.image.image_size
        self.patch_size = config.image.patch_size
        self.num_patches_per_side = config.image.num_patches_per_side
        self.num_patches = config.image.num_patches
        self.pixels_per_patch = config.image.pixels_per_patch
        
        self.value_step = config.quant.value_step
        self.num_values = config.quant.num_values
        
        self.vocab_size = config.vocab_size
        self.bos_token_id = config.model.bos_token_id
        self.eos_token_id = config.model.eos_token_id
        
        # 预计算进制权重 (用于编码/解码)
        # token_id = sum(pixel_idx[i] * base^(n-1-i))
        self.base = self.num_values
        self.powers = torch.tensor(
            [self.base ** (self.pixels_per_patch - 1 - i) for i in range(self.pixels_per_patch)],
            dtype=torch.long
        )
    
    def _quantize_pixel(self, pixel_value: torch.Tensor) -> torch.Tensor:
        """
        量化像素值到离散索引
        pixel_value: [0, 1] -> pixel_idx: {0, 1, 2, 3, 4}
        """
        # 四舍五入到最近的离散值
        pixel_idx = torch.round(pixel_value / self.value_step).long()
        # 确保在有效范围内
        pixel_idx = torch.clamp(pixel_idx, 0, self.num_values - 1)
        return pixel_idx
    
    def _dequantize_pixel(self, pixel_idx: torch.Tensor) -> torch.Tensor:
        """
        反量化像素索引到像素值
        pixel_idx: {0, 1, 2, 3, 4} -> pixel_value: [0, 1]
        """
        return pixel_idx.float() * self.value_step
    
    def _patch_to_token(self, patch_pixels: torch.Tensor) -> torch.Tensor:
        """
        将patch像素转换为token ID
        patch_pixels: (batch, pixels_per_patch) 像素值 [0, 1]
        返回: (batch,) token IDs
        """
        # 量化像素
        pixel_indices = self._quantize_pixel(patch_pixels)  # (batch, pixels_per_patch)
        
        # 编码为token ID (进制转换)
        # token_id = px0_idx * base^3 + px1_idx * base^2 + px2_idx * base^1 + px3_idx * base^0
        powers = self.powers.to(pixel_indices.device)
        token_ids = (pixel_indices * powers).sum(dim=-1)
        
        return token_ids
    
    def _token_to_patch(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        将token ID转换为patch像素
        token_ids: (batch,) token IDs
        返回: (batch, pixels_per_patch) 像素值 [0, 1]
        """
        batch_size = token_ids.shape[0]
        device = token_ids.device
        
        # 解码token ID (进制转换的逆过程)
        pixel_indices = torch.zeros(batch_size, self.pixels_per_patch, dtype=torch.long, device=device)
        
        remaining = token_ids.clone()
        for i in range(self.pixels_per_patch):
            power = self.base ** (self.pixels_per_patch - 1 - i)
            pixel_indices[:, i] = remaining // power
            remaining = remaining % power
        
        # 反量化
        pixel_values = self._dequantize_pixel(pixel_indices)
        
        return pixel_values
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        将图像编码为token序列
        
        序列格式: [BOS, patch_0, patch_1, ..., patch_N-1, EOS]
        
        Args:
            images: (batch, channels, height, width) 或 (channels, height, width)
                    像素值范围 [0, 1]
        
        Returns:
            token_ids: (batch, seq_len) token序列，包含 BOS 和 EOS
        """
        # 处理单张图像
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        batch_size = images.shape[0]
        device = images.device
        
        # 确保是单通道
        if images.shape[1] != 1:
            raise ValueError(f"Expected 1 channel, got {images.shape[1]}")
        
        # 去掉通道维度
        images = images.squeeze(1)  # (batch, H, W)
        
        # 分割成patches
        # (batch, H, W) -> (batch, num_patches, pixels_per_patch)
        patches = self._image_to_patches(images)
        
        # 每个patch编码为token
        # (batch, num_patches, pixels_per_patch) -> (batch, num_patches)
        token_ids = []
        for i in range(self.num_patches):
            patch_tokens = self._patch_to_token(patches[:, i, :])
            token_ids.append(patch_tokens)
        token_ids = torch.stack(token_ids, dim=1)  # (batch, num_patches)
        
        # 添加 BOS 和 EOS (强制)
        bos = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        eos = torch.full((batch_size, 1), self.eos_token_id, dtype=torch.long, device=device)
        token_ids = torch.cat([bos, token_ids, eos], dim=1)
        
        return token_ids
    
    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        将token序列解码为图像
        
        自动移除 BOS/EOS，支持可变长度序列 (EOS 主动终止):
        - 如果序列长度 < num_patches，用 0 填充剩余部分
        - 如果序列长度 > num_patches，截断多余部分
        
        Args:
            token_ids: (batch, seq_len) 或 (seq_len,) token序列
        
        Returns:
            images: (batch, 1, height, width) 像素值范围 [0, 1]
        """
        # 处理单个序列
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        
        batch_size = token_ids.shape[0]
        device = token_ids.device
        
        # 为每个样本独立处理特殊token
        processed_tokens = []
        for i in range(batch_size):
            seq = token_ids[i]
            
            # 移除 BOS (在开头)
            seq = seq[1:]
            
            # 找到第一个 EOS 并截断
            eos_positions = (seq == self.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                first_eos = eos_positions[0].item()
                seq = seq[:first_eos]
            
            processed_tokens.append(seq)
        
        # 统一长度：填充或截断到 num_patches
        padded_tokens = torch.zeros(batch_size, self.num_patches, dtype=torch.long, device=device)
        
        for i, seq in enumerate(processed_tokens):
            seq_len = min(len(seq), self.num_patches)
            padded_tokens[i, :seq_len] = seq[:seq_len]
            # 剩余部分保持为 0 (黑色像素)
        
        token_ids = padded_tokens
        
        # 每个token解码为patch
        patches = []
        for i in range(self.num_patches):
            patch_pixels = self._token_to_patch(token_ids[:, i])
            patches.append(patch_pixels)
        patches = torch.stack(patches, dim=1)  # (batch, num_patches, pixels_per_patch)
        
        # 重组为图像
        images = self._patches_to_image(patches)  # (batch, H, W)
        
        # 添加通道维度
        images = images.unsqueeze(1)  # (batch, 1, H, W)
        
        return images
    
    def _image_to_patches(self, images: torch.Tensor) -> torch.Tensor:
        """
        将图像分割成patches
        
        Args:
            images: (batch, H, W)
        
        Returns:
            patches: (batch, num_patches, pixels_per_patch)
        """
        batch_size = images.shape[0]
        device = images.device
        
        patches = []
        for row in range(self.num_patches_per_side):
            for col in range(self.num_patches_per_side):
                # 提取patch区域
                h_start = row * self.patch_size
                h_end = h_start + self.patch_size
                w_start = col * self.patch_size
                w_end = w_start + self.patch_size
                
                patch = images[:, h_start:h_end, w_start:w_end]  # (batch, patch_size, patch_size)
                patch = patch.reshape(batch_size, -1)  # (batch, pixels_per_patch)
                patches.append(patch)
        
        patches = torch.stack(patches, dim=1)  # (batch, num_patches, pixels_per_patch)
        return patches
    
    def _patches_to_image(self, patches: torch.Tensor) -> torch.Tensor:
        """
        将patches重组为图像
        
        Args:
            patches: (batch, num_patches, pixels_per_patch)
        
        Returns:
            images: (batch, H, W)
        """
        batch_size = patches.shape[0]
        device = patches.device
        
        images = torch.zeros(batch_size, self.image_size, self.image_size, device=device)
        
        patch_idx = 0
        for row in range(self.num_patches_per_side):
            for col in range(self.num_patches_per_side):
                h_start = row * self.patch_size
                h_end = h_start + self.patch_size
                w_start = col * self.patch_size
                w_end = w_start + self.patch_size
                
                patch = patches[:, patch_idx, :].reshape(batch_size, self.patch_size, self.patch_size)
                images[:, h_start:h_end, w_start:w_end] = patch
                patch_idx += 1
        
        return images
    
    def get_position_ids(self, seq_len: int, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取2D位置ID (用于2D RoPE)
        
        Args:
            seq_len: 序列长度 (包括BOS)
            device: 设备
        
        Returns:
            row_ids: (seq_len,) 行位置
            col_ids: (seq_len,) 列位置
        """
        row_ids = []
        col_ids = []
        
        for i in range(seq_len):
            if i == 0:
                # BOS token: 特殊位置 (0, 0) 或者可以用 (-1, -1) 表示
                row_ids.append(0)
                col_ids.append(0)
            else:
                # Patch tokens
                patch_idx = i - 1  # 减去BOS
                if patch_idx < self.num_patches:
                    row = patch_idx // self.num_patches_per_side
                    col = patch_idx % self.num_patches_per_side
                    row_ids.append(row)
                    col_ids.append(col)
                else:
                    # EOS token: 使用最后一个位置
                    row_ids.append(self.num_patches_per_side - 1)
                    col_ids.append(self.num_patches_per_side - 1)
        
        row_ids = torch.tensor(row_ids, dtype=torch.long, device=device)
        col_ids = torch.tensor(col_ids, dtype=torch.long, device=device)
        
        return row_ids, col_ids


if __name__ == "__main__":
    # 测试tokenizer
    config = get_default_config()
    tokenizer = SpriteTokenizer(config)
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"BOS token ID: {tokenizer.bos_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    
    # 创建随机图像
    batch_size = 2
    images = torch.rand(batch_size, 1, 32, 32)
    
    # 编码
    tokens = tokenizer.encode(images)
    print(f"Encoded tokens shape: {tokens.shape}")
    print(f"Token range: [{tokens.min().item()}, {tokens.max().item()}]")
    
    # 解码
    decoded = tokenizer.decode(tokens)
    print(f"Decoded images shape: {decoded.shape}")
    
    # 检查重建误差 (由于量化会有一些误差)
    error = (images - decoded).abs().mean()
    print(f"Reconstruction error: {error.item():.4f}")
    
    # 测试位置ID
    row_ids, col_ids = tokenizer.get_position_ids(tokens.shape[1])
    print(f"Row IDs: {row_ids[:10]}...")
    print(f"Col IDs: {col_ids[:10]}...")
