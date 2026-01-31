"""
Causal Self-Attention with 2D RoPE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..config import SpriteConfig
from .rotary_embedding import RotaryEmbedding2D


class CausalSelfAttention(nn.Module):
    """
    因果自注意力层，带2D RoPE
    """
    
    def __init__(self, config: SpriteConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.model.n_heads
        self.d_model = config.model.d_model
        self.head_dim = config.model.head_dim
        
        assert self.d_model % self.n_heads == 0
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # 输出投影
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # 2D RoPE
        self.rotary_emb = RotaryEmbedding2D(
            dim=self.head_dim,
            max_position=config.image.num_patches_per_side + 1  # +1 for safety
        )
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.model.dropout)
        self.resid_dropout = nn.Dropout(config.model.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            row_ids: (seq_len,) 行位置
            col_ids: (seq_len,) 列位置
            attention_mask: (batch, 1, seq_len, seq_len) 或 None
            use_cache: 是否使用KV cache
            past_kv: 缓存的 (K, V)
        
        Returns:
            output: (batch, seq_len, d_model)
            present_kv: 新的 (K, V) cache
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape为多头形式
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # 应用2D RoPE到Q和K
        q = self.rotary_emb(q, row_ids, col_ids)
        k = self.rotary_emb(k, row_ids, col_ids)
        
        # 处理KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        present_kv = (k, v) if use_cache else None
        
        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        # (batch, n_heads, seq_len, head_dim) @ (batch, n_heads, head_dim, kv_len)
        # -> (batch, n_heads, seq_len, kv_len)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # 应用因果mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            # 创建因果mask
            kv_len = k.shape[2]
            causal_mask = torch.triu(
                torch.full((seq_len, kv_len), float("-inf"), device=x.device),
                diagonal=kv_len - seq_len + 1
            )
            attn_weights = attn_weights + causal_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 计算输出
        # (batch, n_heads, seq_len, kv_len) @ (batch, n_heads, kv_len, head_dim)
        # -> (batch, n_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back: (batch, seq_len, n_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Reshape: (batch, seq_len, d_model)
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output, present_kv
