"""
Transformer Decoder Block
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..config import SpriteConfig
from .rms_norm import RMSNorm
from .attention import CausalSelfAttention
from .feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """Transformer Decoder Block"""
    
    def __init__(self, config: SpriteConfig):
        super().__init__()
        
        self.attention = CausalSelfAttention(config)
        self.feed_forward = FeedForward(config)
        
        self.attention_norm = RMSNorm(config.model.d_model)
        self.ffn_norm = RMSNorm(config.model.d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm + Attention + Residual
        residual = x
        x = self.attention_norm(x)
        x, present_kv = self.attention(x, row_ids, col_ids, attention_mask, use_cache, past_kv)
        x = residual + x
        
        # Pre-norm + FFN + Residual
        residual = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        x = residual + x
        
        return x, present_kv
