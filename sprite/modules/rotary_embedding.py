"""
2D Rotary Position Embedding (2D RoPE)
将 head_dim 分成两半，分别编码行位置和列位置
"""

import torch
import torch.nn as nn


class RotaryEmbedding2D(nn.Module):
    """
    2D旋转位置编码 (2D RoPE)
    
    将head_dim分成两半，分别编码行位置和列位置
    """
    
    def __init__(self, dim: int, max_position: int = 64, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.max_position = max_position
        self.base = base
        
        # 预计算频率
        # freqs = 1 / (base^(2k/dim)), k = 0, 1, ..., dim/4 - 1
        quarter_dim = self.half_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, quarter_dim, dtype=torch.float32) / quarter_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 预计算所有位置的sin/cos
        self._build_cache(max_position)
    
    def _build_cache(self, max_position: int):
        """预计算位置编码缓存"""
        positions = torch.arange(max_position, dtype=torch.float32)
        
        # (max_position, quarter_dim)
        freqs = torch.outer(positions, self.inv_freq)
        
        # (max_position, half_dim) - 每个频率重复两次用于复数旋转
        freqs = torch.cat([freqs, freqs], dim=-1)
        
        # 缓存cos和sin
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        row_ids: torch.Tensor, 
        col_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        对输入应用2D RoPE
        
        Args:
            x: (batch, seq_len, n_heads, head_dim)
            row_ids: (seq_len,) 行位置索引
            col_ids: (seq_len,) 列位置索引
        
        Returns:
            旋转后的x: (batch, seq_len, n_heads, head_dim)
        """
        seq_len = x.shape[1]
        
        # 获取行和列的cos/sin
        # (seq_len, half_dim)
        cos_row = self.cos_cached[row_ids]
        sin_row = self.sin_cached[row_ids]
        cos_col = self.cos_cached[col_ids]
        sin_col = self.sin_cached[col_ids]
        
        # 扩展维度以匹配x
        # (1, seq_len, 1, half_dim)
        cos_row = cos_row.unsqueeze(0).unsqueeze(2)
        sin_row = sin_row.unsqueeze(0).unsqueeze(2)
        cos_col = cos_col.unsqueeze(0).unsqueeze(2)
        sin_col = sin_col.unsqueeze(0).unsqueeze(2)
        
        # 分割x为行和列部分
        x_row = x[..., :self.half_dim]  # (batch, seq_len, n_heads, half_dim)
        x_col = x[..., self.half_dim:]  # (batch, seq_len, n_heads, half_dim)
        
        # 对行部分应用旋转
        x_row_rotated = self._apply_rotary(x_row, cos_row, sin_row)
        
        # 对列部分应用旋转
        x_col_rotated = self._apply_rotary(x_col, cos_col, sin_col)
        
        # 合并
        return torch.cat([x_row_rotated, x_col_rotated], dim=-1)
    
    def _apply_rotary(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> torch.Tensor:
        """
        应用旋转变换
        
        对于每对相邻维度 (x0, x1):
        x0' = x0 * cos - x1 * sin
        x1' = x0 * sin + x1 * cos
        """
        # 将x reshape为复数形式处理
        # (batch, seq_len, n_heads, half_dim) -> (batch, seq_len, n_heads, half_dim/2, 2)
        x_reshape = x.reshape(*x.shape[:-1], -1, 2)
        
        # 分离实部和虚部
        x0 = x_reshape[..., 0]  # (batch, seq_len, n_heads, half_dim/2)
        x1 = x_reshape[..., 1]
        
        # cos和sin也需要reshape
        cos = cos[..., ::2]  # 取偶数索引
        sin = sin[..., ::2]
        
        # 应用旋转
        x0_rot = x0 * cos - x1 * sin
        x1_rot = x0 * sin + x1 * cos
        
        # 重新组合
        x_rot = torch.stack([x0_rot, x1_rot], dim=-1)
        return x_rot.reshape(*x.shape)
