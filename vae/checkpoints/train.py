"""
VAE训练脚本
支持KL退火、学习率调度、早停、梯度监控等高级特性
"""
import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from tqdm import tqdm

from model import (
    VAE, ConvVAE, DeepConvVAE,
    vae_loss, conv_vae_loss
)


class KLAnnealer:
    """KL散度退火调度器

    为什么需要KL退火？
    - 防止"后验坍塌"：模型忽略潜在变量，退化为自编码器
    - 逐渐增加KL约束，让模型先学好重构，再学习规则的潜在分布

    数学原理：
    - KL散度作为正则化项，如果一开始权重太大，模型会直接学习
      将所有样本映射到N(0,1)，导致重构质量很差
    - 通过退火策略，让模型有时间学习有意义的潜在表示
    """

    def __init__(
            self,
            total_steps: int,
            start_step: int = 0,
            anneal_type: str = 'linear',
            min_kl_weight: float = 0.0,
            max_kl_weight: float = 1.0
    ):
        """
        Args:
            total_steps: 总训练步数
            start_step: 开始退火的步数
            anneal_type: 退火类型 ('linear', 'sigmoid', 'cyclical')
            min_kl_weight: 最小KL权重
            max_kl_weight: 最大KL权重
        """
        self.total_steps = total_steps
        self.start_step = start_step
        self.anneal_type = anneal_type
        self.min_kl_weight = min_kl_weight
        self.max_kl_weight = max_kl_weight
        self.current_step = 0

    def step(self) -> float:
        """计算当前KL权重"""
        self.current_step += 1

        if self.current_step < self.start_step:
            return self.min_kl_weight

        progress = (self.current_step - self.start_step) / (self.total_steps - self.start_step)
        progress = min(progress, 1.0)

        if self.anneal_type == 'linear':
            weight = progress
        elif self.anneal_type == 'sigmoid':
            # 平滑的S曲线: 在0.5处快速增长
            weight = float(1 / (1 + torch.exp(torch.tensor(-10 * (progress - 0.5)))))
        elif self.anneal_type == 'cyclical':
            # 周期性退火（用于更好的探索）
            cycle_progress = (progress * 4) % 1.0
            weight = cycle_progress
        else:
            raise ValueError(f"Unknown anneal type: {self.anneal_type}")

        # 缩放到指定范围
        return self.min_kl_weight + weight * (self.max_kl_weight - self.min_kl_weight)


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing学习率调度器

    为什么使用这个策略？
    1. Warmup: 训练初期使用小学习率，防止参数突变
    2. Cosine Annealing: 平滑地降低学习率，帮助收敛到更好的局部最优
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_steps: int,
            total_steps: int,
            min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        """更新学习率"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Warmup阶段: 线性增长
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine Annealing阶段
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                    1 + torch.cos(torch.tensor(progress * 3.14159))
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = float(lr)

        return float(lr)

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class EarlyStopping:
    """早停机制

    当验证集性能不再提升时，提前终止训练
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改进量
            mode: 'min'表示越小越好，'max'表示越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # 检查是否有改进
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class GradientMonitor:
    """梯度监控器

    帮助诊断训练问题:
    - 梯度消失: 梯度太小，网络难以学习
    - 梯度爆炸: 梯度太大，导致训练不稳定
    """

    def __init__(self):
        self.grad_norms = []

    def update(self, model: torch.nn.Module):
        """记录当前梯度范数"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)

    def get_stats(self) -> Dict[str, float]:
        """获取梯度统计信息"""
        if not self.grad_norms:
            return {}

        import numpy as np
        norms = np.array(self.grad_norms)
        return {
            'grad_norm_mean': float(np.mean(norms)),
            'grad_norm_std': float(np.std(norms)),
            'grad_norm_max': float(np.max(norms)),
            'grad_norm_min': float(np.min(norms))
        }

    def reset(self):
        """重置统计"""
        self.grad_norms = []


class Trainer:
    """VAE训练器 - 优化版本"""

    def __init__(
            self,
            model: torch.nn.Module,
            model_type: str,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            device: str = 'cuda',
            lr: float = 1e-3,
            weight_decay: float = 1e-5,
            kl_anneal_epochs: int = 50,
            warmup_epochs: int = 5,
            early_stopping_patience: int = 15,
            gradient_clip_norm: float = 5.0,
            save_dir: str = './checkpoints',
            log_dir: str = './runs'
    ):
        """
        Args:
            model: VAE模型
            model_type: 模型类型 ('vae', 'conv_vae', 'deep_conv_vae')
        """
        self.model = model.to(device)
        self.model_type = model_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_clip_norm = gradient_clip_norm

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        total_steps = len(train_loader) * 100  # 假设最多训练100个epoch
        warmup_steps = len(train_loader) * warmup_epochs
        self.lr_scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )

        # KL退火
        total_steps = kl_anneal_epochs * len(train_loader)
        self.kl_annealer = KLAnnealer(
            total_steps=total_steps,
            anneal_type='linear'
        )

        # 早停
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=1e-4
        )

        # 梯度监控
        self.grad_monitor = GradientMonitor()

        # 日志和保存
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []

    def _compute_loss(self, recon_x, x, mu, logvar, kl_weight):
        """根据模型类型计算损失"""
        if self.model_type == 'vae':
            return vae_loss(recon_x, x, mu, logvar, kl_weight)
        else:
            return conv_vae_loss(recon_x, x, mu, logvar, kl_weight)

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.grad_monitor.reset()

        epoch_losses = {
            'total': 0.0,
            'recon': 0.0,
            'kl': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)

            # 前向传播
            recon_images, mu, logvar = self.model(images)

            # 计算损失（带KL退火）
            kl_weight = self.kl_annealer.step()
            loss, loss_dict = self._compute_loss(
                recon_images, images, mu, logvar, kl_weight
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度监控
            self.grad_monitor.update(self.model)

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip_norm
            )

            self.optimizer.step()

            # 学习率调度
            current_lr = self.lr_scheduler.step()

            # 记录损失
            epoch_losses['total'] += loss_dict['total_loss']
            epoch_losses['recon'] += loss_dict['recon_loss']
            epoch_losses['kl'] += loss_dict['kl_loss']

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'recon': f"{loss_dict['recon_loss']:.4f}",
                'kl': f"{loss_dict['kl_loss']:.4f}",
                'kl_w': f"{kl_weight:.4f}",
                'lr': f"{current_lr:.6f}"
            })

            # TensorBoard日志
            if self.global_step % 100 == 0:
                self.writer.add_scalar('train/loss', loss_dict['total_loss'], self.global_step)
                self.writer.add_scalar('train/recon_loss', loss_dict['recon_loss'], self.global_step)
                self.writer.add_scalar('train/kl_loss', loss_dict['kl_loss'], self.global_step)
                self.writer.add_scalar('train/kl_weight', kl_weight, self.global_step)
                self.writer.add_scalar('train/learning_rate', current_lr, self.global_step)

            self.global_step += 1

        # 计算平均损失
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # 记录梯度统计
        grad_stats = self.grad_monitor.get_stats()
        if grad_stats:
            for key, value in grad_stats.items():
                self.writer.add_scalar(f'train/{key}', value, self.current_epoch)

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}

        self.model.eval()

        val_losses = {
            'total': 0.0,
            'recon': 0.0,
            'kl': 0.0
        }

        for images, _ in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)

            recon_images, mu, logvar = self.model(images)
            loss, loss_dict = self._compute_loss(
                recon_images, images, mu, logvar, kl_weight=1.0
            )

            val_losses['total'] += loss_dict['total_loss']
            val_losses['recon'] += loss_dict['recon_loss']
            val_losses['kl'] += loss_dict['kl_loss']

        # 计算平均损失
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches

        return val_losses

    def save_checkpoint(self, is_best: bool = False, epoch: Optional[int] = None):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'model_type': self.model_type
        }

        # 保存最新检查点
        checkpoint_path = self.save_dir / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = self.save_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with val_loss={self.best_val_loss:.4f}")

        # 定期保存epoch检查点
        if epoch is not None and epoch % 20 == 0:
            epoch_path = self.save_dir / f'epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")

    @torch.no_grad()
    def log_samples(self, num_samples: int = 16):
        """记录生成样本到TensorBoard"""
        self.model.eval()

        # 随机生成
        samples = self.model.sample(num_samples, self.device)

        if self.model_type == 'vae':
            samples = samples.view(-1, 1, 28, 28)
        else:
            # ConvVAE和DeepConvVAE已经是正确的形状
            pass

        # 制作网格
        grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
        self.writer.add_image('samples/generated', grid, self.current_epoch)

        # 重构样本
        if self.val_loader is not None:
            real_images, _ = next(iter(self.val_loader))
            real_images = real_images[:num_samples].to(self.device)
            recon_images, _, _ = self.model(real_images)

            if self.model_type == 'vae':
                recon_images = recon_images.view(-1, 1, 28, 28)

            # 对比真实图像和重构图像
            comparison = torch.cat([real_images, recon_images])
            grid = torchvision.utils.make_grid(comparison, nrow=num_samples, normalize=True)
            self.writer.add_image('samples/reconstruction', grid, self.current_epoch)

    def train(self, num_epochs: int):
        """完整训练流程"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model type: {self.model_type}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 60)

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # 训练
            train_losses = self.train_epoch()

            # 验证
            val_losses = self.validate()

            # 记录历史
            epoch_stats = {
                'epoch': epoch,
                'train': train_losses,
                'val': val_losses
            }
            self.training_history.append(epoch_stats)

            # 日志
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_losses['total']:.4f} "
                  f"(Recon: {train_losses['recon']:.4f}, KL: {train_losses['kl']:.4f})")

            if val_losses:
                print(f"  Val Loss: {val_losses['total']:.4f} "
                      f"(Recon: {val_losses['recon']:.4f}, KL: {val_losses['kl']:.4f})")

                self.writer.add_scalar('val/loss', val_losses['total'], epoch)
                self.writer.add_scalar('val/recon_loss', val_losses['recon'], epoch)
                self.writer.add_scalar('val/kl_loss', val_losses['kl'], epoch)

            # 保存检查点
            is_best = False
            if val_losses and val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                is_best = True

            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(is_best, epoch)

            # 记录生成样本
            if epoch % 5 == 0:
                self.log_samples()

            # 早停检查
            if val_losses and self.early_stopping(val_losses['total']):
                print(f"\n⚠ Early stopping triggered at epoch {epoch}")
                print(f"  No improvement for {self.early_stopping.patience} epochs")
                break

        print("\n✓ Training completed!")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        print(f"  Total epochs: {self.current_epoch + 1}")
        self.writer.close()

        return self.training_history


def get_data_loaders(
        batch_size: int = 128,
        val_split: float = 0.1,
        num_workers: int = 4
):
    """获取数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 注意: 不需要Normalize，因为VAE需要[0,1]范围的数据
    ])

    # 完整训练集
    full_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 划分训练集和验证集
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader


def create_model(model_type: str, latent_dim: int, **kwargs) -> torch.nn.Module:
    """创建模型

    Args:
        model_type: 'vae', 'conv_vae', 'deep_conv_vae'
        latent_dim: 潜在空间维度
    """
    if model_type == 'vae':
        hidden_dims = kwargs.get('hidden_dims', [512, 256])
        model = VAE(
            input_dim=784,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim
        )
    elif model_type == 'conv_vae':
        hidden_channels = kwargs.get('hidden_channels', [32, 64])
        model = ConvVAE(
            in_channels=1,
            image_size=28,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim
        )
    elif model_type == 'deep_conv_vae':
        hidden_channels = kwargs.get('hidden_channels', [32, 64, 128])
        num_res_blocks = kwargs.get('num_res_blocks', 2)
        model = DeepConvVAE(
            in_channels=1,
            image_size=28,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train VAE on MNIST - Optimized')

    # 模型参数
    parser.add_argument('--model-type', type=str, default='deep_conv_vae',
                        choices=['vae', 'conv_vae', 'deep_conv_vae'],
                        help='模型类型')
    parser.add_argument('--latent-dim', type=int, default=32, help='潜在空间维度')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 256],
                        help='VAE隐藏层维度')
    parser.add_argument('--hidden-channels', type=int, nargs='+', default=[32, 64, 128],
                        help='ConvVAE隐藏通道数')
    parser.add_argument('--num-res-blocks', type=int, default=2,
                        help='DeepConvVAE残差块数量')

    # 训练参数
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--kl-anneal-epochs', type=int, default=30,
                        help='KL退火轮数')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='学习率预热轮数')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                        help='早停patience')
    parser.add_argument('--gradient-clip', type=float, default=5.0,
                        help='梯度裁剪阈值')
    # 数据参数
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')

    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='模型保存路径')
    parser.add_argument('--log-dir', type=str, default='./runs',
                        help='日志路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 数据加载器
    train_loader, val_loader = get_data_loaders(
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers
    )
    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Val samples: {len(val_loader.dataset):,}")

    # 创建模型
    model = create_model(
        model_type=args.model_type,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        hidden_channels=args.hidden_channels,
        num_res_blocks=args.num_res_blocks
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        model_type=args.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        kl_anneal_epochs=args.kl_anneal_epochs,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.early_stopping_patience,
        gradient_clip_norm=args.gradient_clip,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    history = trainer.train(args.epochs)

    # 保存配置和历史
    config = vars(args)
    config_path = Path(args.save_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    history_path = Path(args.save_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Configuration saved to {config_path}")
    print(f"✓ Training history saved to {history_path}")

# python train.py \
#     --model-type deep_conv_vae \
#     --latent-dim 32 \
#     --batch-size 128 \
#     --epochs 100
if __name__ == '__main__':
    main()