"""
Sprite Image LLM - 模型定义
基于Transformer Decoder的自回归图像生成模型，使用2D RoPE位置编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import SpriteConfig, get_default_config
from .modules import RMSNorm, TransformerBlock


class SpriteModel(nn.Module):
    """
    Sprite图像生成模型
    
    基于自回归Transformer的图像生成，使用2D RoPE位置编码
    """
    
    def __init__(self, config: SpriteConfig = None):
        super().__init__()
        
        if config is None:
            config = get_default_config()
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.model.d_model
        self.n_layers = config.model.n_layers
        
        # Token Embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(self.n_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(self.d_model)
        
        # Output head (language modeling head)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[list] = None
    ) -> dict:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            row_ids: (seq_len,) 行位置
            col_ids: (seq_len,) 列位置
            attention_mask: 注意力mask
            labels: (batch, seq_len) 用于计算loss的标签
            use_cache: 是否使用KV cache
            past_key_values: 之前的KV cache
        
        Returns:
            dict with:
                - logits: (batch, seq_len, vocab_size)
                - loss: scalar (if labels provided)
                - past_key_values: KV cache (if use_cache)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embedding
        hidden_states = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        
        # 处理past_key_values
        if past_key_values is None:
            past_key_values = [None] * self.n_layers
        
        present_key_values = []
        
        # 通过所有Transformer层
        for i, layer in enumerate(self.layers):
            hidden_states, present_kv = layer(
                hidden_states,
                row_ids,
                col_ids,
                attention_mask,
                use_cache,
                past_key_values[i]
            )
            if use_cache:
                present_key_values.append(present_kv)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)
        
        # 计算loss
        loss = None
        if labels is not None:
            # labels 已经在 Dataset 中做过 shift，直接计算
            # input_ids: [BOS, p1, ..., pN]  (不含 EOS)
            # labels:    [p1, ..., pN, EOS]  (不含 BOS)
            logits_flat = logits.view(-1, self.vocab_size)
            labels_flat = labels.view(-1)
            
            # Cross entropy
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
        
        result = {"logits": logits}
        if loss is not None:
            result["loss"] = loss
        if use_cache:
            result["past_key_values"] = present_key_values
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        tokenizer,
        num_samples: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        device: torch.device = None,
        max_tokens: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成图像 (EOS 主动终止)
        
        Args:
            tokenizer: SpriteTokenizer实例
            num_samples: 生成数量
            temperature: 采样温度
            top_k: top-k采样
            top_p: nucleus采样
            device: 设备
            max_tokens: 最大生成token数 (不含BOS)，默认为 num_patches + 1
        
        Returns:
            images: (num_samples, 1, H, W) 生成的图像
            token_sequences: (num_samples, seq_len) 生成的token序列
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        # 初始化序列，只有BOS
        generated = torch.full(
            (num_samples, 1),
            tokenizer.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # 最大生成长度 (允许生成 EOS)
        if max_tokens is None:
            max_tokens = tokenizer.num_patches + 1  # patches + EOS
        
        # 追踪每个样本是否已结束
        finished = torch.zeros(num_samples, dtype=torch.bool, device=device)
        
        for step in range(max_tokens):
            # 如果所有样本都已生成 EOS，提前退出
            if finished.all():
                break
            
            # 获取位置ID
            seq_len = generated.shape[1]
            row_ids, col_ids = tokenizer.get_position_ids(seq_len, device)
            
            # Forward pass
            outputs = self.forward(generated, row_ids, col_ids)
            logits = outputs["logits"]
            
            # 只取最后一个位置的logits
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)
            
            # 对已经结束的样本，强制生成 PAD (用 EOS 代替)
            # 这样保证所有序列长度一致
            next_token_logits[finished] = float("-inf")
            next_token_logits[finished, tokenizer.eos_token_id] = 0.0
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Top-k 采样
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Top-p (nucleus) 采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")
            
            # 采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # 更新结束状态
            finished = finished | (next_token.squeeze(-1) == tokenizer.eos_token_id)
            
            # 追加到序列
            generated = torch.cat([generated, next_token], dim=1)
        
        # 解码为图像
        images = tokenizer.decode(generated)
        
        return images, generated
    
    def get_num_params(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # 测试模型
    from .tokenizer import SpriteTokenizer
    
    config = get_default_config()
    model = SpriteModel(config)
    tokenizer = SpriteTokenizer(config)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # 测试forward
    batch_size = 2
    images = torch.rand(batch_size, 1, 32, 32)
    tokens = tokenizer.encode(images)
    
    row_ids, col_ids = tokenizer.get_position_ids(tokens.shape[1])
    
    outputs = model(tokens, row_ids, col_ids, labels=tokens)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # 测试生成
    generated_images, generated_tokens = model.generate(tokenizer, num_samples=1)
    print(f"Generated images shape: {generated_images.shape}")
    print(f"Generated tokens shape: {generated_tokens.shape}")