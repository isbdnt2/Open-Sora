"""
Sprite Image LLM - 训练器
负责模型训练逻辑
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.amp import autocast, GradScaler
from typing import Optional, Dict, Any
from pathlib import Path
import json
from tqdm import tqdm

from .config import SpriteConfig, get_default_config
from .model import SpriteModel
from .tokenizer import SpriteTokenizer
from .dataset import create_dataloader


class Trainer:
    """
    训练器
    
    负责模型训练、验证、保存和日志记录
    """
    
    def __init__(
        self,
        model: SpriteModel,
        tokenizer: SpriteTokenizer,
        config: SpriteConfig = None,
        train_dataloader=None,
        eval_dataloader=None,
        device: Optional[torch.device] = None,
        use_amp: bool = True
    ):
        if config is None:
            config = get_default_config()
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # 混合精度
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = None
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        
        # 日志
        self.log_history = []
        
        # 创建输出目录
        self.output_dir = Path(config.train.output_dir)
        self.checkpoint_dir = Path(config.train.checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> AdamW:
        """创建优化器，支持权重衰减"""
        # 不对embedding和LayerNorm应用权重衰减
        no_decay = ["bias", "norm", "embedding"]
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n.lower() for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.train.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n.lower() for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.train.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
    
    def _create_scheduler(self, num_training_steps: int):
        """创建学习率调度器，带warmup"""
        warmup_steps = self.config.train.warmup_steps
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Warmup阶段：线性增加
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine衰减
                progress = float(current_step - warmup_steps) / float(
                    max(1, num_training_steps - warmup_steps)
                )
                return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
    
    def train(
        self,
        num_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        resume_from: Optional[str] = None
    ):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            max_steps: 最大训练步数
            resume_from: 从checkpoint恢复
        """
        if num_epochs is None:
            num_epochs = self.config.train.epochs
        
        if self.train_dataloader is None:
            raise ValueError("train_dataloader is required for training")
        
        # 恢复训练
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # 计算总训练步数
        steps_per_epoch = len(self.train_dataloader)
        total_steps = num_epochs * steps_per_epoch
        if max_steps:
            total_steps = min(total_steps, max_steps)
        
        # 创建调度器
        self._create_scheduler(total_steps)
        
        print(f"Starting training...")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps}")
        print(f"  Batch size: {self.config.train.batch_size}")
        print(f"  Learning rate: {self.config.train.learning_rate}")
        print(f"  Model parameters: {self.model.get_num_params():,}")
        print(f"  Mixed precision: {self.use_amp}")
        print()
        
        # 训练循环
        self.model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                dynamic_ncols=True
            )
            
            for batch in progress_bar:
                if max_steps and self.global_step >= max_steps:
                    break
                
                loss = self._train_step(batch)
                running_loss += loss
                epoch_loss += loss
                epoch_steps += 1
                self.global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
                # 日志
                if self.global_step % self.config.train.log_interval == 0:
                    avg_loss = running_loss / self.config.train.log_interval
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed
                    
                    log_entry = {
                        "step": self.global_step,
                        "epoch": epoch + 1,
                        "loss": avg_loss,
                        "lr": self.scheduler.get_last_lr()[0],
                        "steps_per_sec": steps_per_sec
                    }
                    self.log_history.append(log_entry)
                    running_loss = 0.0
                
            # Epoch结束
            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            if max_steps and self.global_step >= max_steps:
                break
        
        print("\nTraining completed!")
        self.save_checkpoint("final")
        self._save_log_history()
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """单步训练"""
        # 移动到设备
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # 获取位置ID
        seq_len = input_ids.shape[1]
        row_ids, col_ids = self.tokenizer.get_position_ids(seq_len, self.device)
        
        # 前向传播
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with autocast(device_type='cuda'):
                outputs = self.model(input_ids, row_ids, col_ids, labels=labels)
                loss = outputs["loss"]
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.train.max_grad_norm
            )
            
            # 优化器步进
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(input_ids, row_ids, col_ids, labels=labels)
            loss = outputs["loss"]
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.train.max_grad_norm
            )
            
            self.optimizer.step()
        
        # 更新学习率 (在 optimizer.step() 之后)
        if self.scheduler:
            self.scheduler.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """评估模型"""
        if self.eval_dataloader is None:
            raise ValueError("eval_dataloader is required for evaluation")
        
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            seq_len = input_ids.shape[1]
            row_ids, col_ids = self.tokenizer.get_position_ids(seq_len, self.device)
            
            if self.use_amp:
                with autocast(device_type='cuda'):
                    outputs = self.model(input_ids, row_ids, col_ids, labels=labels)
            else:
                outputs = self.model(input_ids, row_ids, col_ids, labels=labels)
            
            total_loss += outputs["loss"].item()
            total_steps += 1
        
        return total_loss / total_steps
    
    def save_checkpoint(self, name: str):
        """保存checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": {
                "image": vars(self.config.image),
                "quant": vars(self.config.quant),
                "model": vars(self.config.model),
                "train": vars(self.config.train),
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        """加载checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if checkpoint.get("scaler_state_dict") and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        
        print(f"Checkpoint loaded: {path}")
        print(f"  Resuming from step {self.global_step}, epoch {self.epoch}")
    
    def _save_log_history(self):
        """保存训练日志"""
        log_path = self.output_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.log_history, f, indent=2)
        print(f"Training log saved: {log_path}")


def train(
    config: SpriteConfig = None,
    image_dir: Optional[str] = None,
    use_synthetic: bool = True,
    synthetic_size: int = 10000,
    num_epochs: int = 10,
    device: Optional[str] = None,
    resume_from: Optional[str] = None
):
    """
    便捷的训练函数
    """
    if config is None:
        config = get_default_config()
    
    # 创建tokenizer
    tokenizer = SpriteTokenizer(config)
    
    # 创建模型
    model = SpriteModel(config)
    
    # 创建数据加载器
    train_dataloader = create_dataloader(
        config=config,
        tokenizer=tokenizer,
        image_dir=image_dir,
        use_synthetic=use_synthetic,
        synthetic_size=synthetic_size,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers
    )
    
    # 设备
    if device:
        device = torch.device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataloader=train_dataloader,
        device=device
    )
    
    # 开始训练
    trainer.train(num_epochs=num_epochs, resume_from=resume_from)
    
    return trainer


if __name__ == "__main__":
    # 快速测试训练
    config = get_default_config()
    
    # 使用较小的配置进行测试
    config.train.batch_size = 8
    config.train.log_interval = 10
    config.train.eval_interval = 50
    config.train.save_interval = 100
    config.train.num_workers = 0
    
    trainer = train(
        config=config,
        use_synthetic=True,
        synthetic_size=100,
        num_epochs=2
    )
