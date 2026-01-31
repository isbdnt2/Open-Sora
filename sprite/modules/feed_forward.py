"""
Feed Forward Network (SwiGLU 风格)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import SpriteConfig


class FeedForward(nn.Module):
    """前馈网络 (SwiGLU风格)"""
    
    def __init__(self, config: SpriteConfig):
        super().__init__()
        self.d_model = config.model.d_model
        self.d_ff = config.model.d_ff
        
        self.w1 = nn.Linear(self.d_model, self.d_ff, bias=False)
        self.w2 = nn.Linear(self.d_ff, self.d_model, bias=False)
        self.w3 = nn.Linear(self.d_model, self.d_ff, bias=False)  # Gate
        
        self.dropout = nn.Dropout(config.model.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w2(SiLU(w1(x)) * w3(x))
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
