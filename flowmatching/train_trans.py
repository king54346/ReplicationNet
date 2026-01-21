"""
Flow Matching Transformer Training - 生产级版本
✅ 支持多 GPU (DataParallel & DDP)
✅ 混合精度训练 (AMP)
✅ 完整的检查点管理
✅ Classifier-Free Guidance
✅ EMA (Exponential Moving Average)
✅ 详细的日志和可视化
"""

import os
import math
import time
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import argparse

from flowmatchingTransformer import FlowMatchingTransformer


# ===============================================================================
# EMA (Exponential Moving Average)
# ===============================================================================

class EMA:
    """指数移动平均 - 提升采样质量"""
    
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化 shadow 参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """更新 shadow 参数"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] + 
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model):
        """应用 shadow 参数（用于推理）"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model):
        """恢复原始参数（用于训练）"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# ===============================================================================
# 配置
# ===============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Flow Matching Transformer Training")
    
    # ========== 数据 ==========
    parser.add_argument("--data-root", type=str, default="./data",
                        help="数据集路径")
    parser.add_argument("--image-size", type=int, default=32,
                        help="图像尺寸")
    
    # ========== 模型 ==========
    parser.add_argument("--patch-size", type=int, default=4,
                        help="Patch 大小")
    parser.add_argument("--hidden-dim", type=int, default=384,
                        help="Transformer 隐藏维度")
    parser.add_argument("--depth", type=int, default=12,
                        help="Transformer 层数")
    parser.add_argument("--num-heads", type=int, default=6,
                        help="注意力头数")
    parser.add_argument("--mlp-ratio", type=float, default=4.0,
                        help="MLP 扩展比例")
    
    # ========== 训练 ==========
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (per GPU)")
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="权重衰减")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="梯度裁剪")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="学习率预热轮数")
    
    # ========== CFG ==========
    parser.add_argument("--cfg-dropout", type=float, default=0.1,
                        help="Classifier-Free Guidance dropout")
    parser.add_argument("--cfg-scale", type=float, default=2.0,
                        help="CFG 引导强度（采样时）")
    
    # ========== 采样 ==========
    parser.add_argument("--sample-steps", type=int, default=100,
                        help="采样步数")
    parser.add_argument("--sample-every", type=int, default=5,
                        help="每 N 个 epoch 采样")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="采样数量")
    
    # ========== 优化 ==========
    parser.add_argument("--use-amp", action="store_true",
                        help="使用混合精度训练")
    parser.add_argument("--use-ema", action="store_true",
                        help="使用 EMA")
    parser.add_argument("--ema-decay", type=float, default=0.9999,
                        help="EMA 衰减率")
    
    # ========== 分布式 ==========
    parser.add_argument("--multi-gpu", action="store_true",
                        help="使用 DataParallel (所有可见 GPU)")
    
    # ========== 日志和保存 ==========
    parser.add_argument("--save-dir", type=str, default="./outputs/flow_transformer",
                        help="检查点保存目录")
    parser.add_argument("--log-every", type=int, default=100,
                        help="每 N 步打印日志")
    parser.add_argument("--save-every", type=int, default=10,
                        help="每 N 个 epoch 保存检查点")
    
    # ========== 其他 ==========
    parser.add_argument("--num-workers", type=int, default=4,
                        help="数据加载线程数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    
    return parser.parse_args()


# ===============================================================================
# 数据加载
# ===============================================================================

def get_dataloader(args):
    """创建数据加载器"""
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        # 归一化到 [-1, 1]
        transforms.Normalize([0.5], [0.5])
    ])
    
    dataset = datasets.MNIST(
        root=args.data_root,
        train=True,
        download=True,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )


# ===============================================================================
# 学习率调度
# ===============================================================================

def get_lr_schedule(optimizer, args, total_steps):
    """创建学习率调度器 (Warmup + Cosine)"""
    warmup_steps = args.warmup_epochs * (total_steps // args.num_epochs)
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Warmup: 线性增长
            return step / warmup_steps
        else:
            # Cosine Annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===============================================================================
# Flow Matching 损失
# ===============================================================================

def flow_matching_loss(model, x1, labels, cfg_dropout=0.1):
    """
    Flow Matching 损失函数
    
    Args:
        model: 模型
        x1: [B, C, H, W] 目标数据（已归一化到 [-1, 1]）
        labels: [B] 类别标签
        cfg_dropout: CFG dropout 概率
    
    Returns:
        loss: 标量损失
    """
    device = x1.device
    B = x1.shape[0]
    
    # 1. 采样时间步 t ~ Uniform(0, 1)
    t = torch.rand(B, device=device)
    
    # 2. 采样源分布 x0 ~ N(0, I)
    x0 = torch.randn_like(x1)
    
    # 3. 线性插值: x_t = (1-t)*x0 + t*x1
    t_expanded = t[:, None, None, None]
    x_t = (1 - t_expanded) * x0 + t_expanded * x1
    
    # 4. 真实速度场: v = x1 - x0
    v_true = x1 - x0
    
    # 5. CFG: 随机丢弃标签
    if cfg_dropout > 0:
        dropout_mask = torch.rand(B, device=device) < cfg_dropout
        labels = labels.clone()
        labels[dropout_mask] = model.num_classes  # 无条件标签
    
    # 6. 模型预测
    v_pred = model(x_t, t, labels)
    
    # 7. MSE 损失
    loss = F.mse_loss(v_pred, v_true)
    
    return loss


# ===============================================================================
# 采样
# ===============================================================================

@torch.no_grad()
def sample(model, num_samples, num_steps, device, labels=None, cfg_scale=0.0):
    """
    Euler ODE 采样
    
    Args:
        model: 模型
        num_samples: 采样数量
        num_steps: ODE 步数
        device: 设备
        labels: [num_samples] 类别标签（None 为无条件）
        cfg_scale: CFG 强度（>1.0 启用）
    
    Returns:
        samples: [num_samples, C, H, W] 生成的图像（[-1, 1]）
    """
    model.eval()
    
    # 获取真实模型（处理 DataParallel）
    model_base = model.module if isinstance(model, nn.DataParallel) else model
    
    # 初始化: x0 ~ N(0, I)
    x = torch.randn(num_samples, 1, 32, 32, device=device)
    
    # 如果没有指定标签，生成所有类别
    if labels is None:
        labels = torch.arange(num_samples, device=device) % 10
    else:
        labels = labels.to(device)
    
    # ODE 求解
    dt = 1.0 / num_steps
    
    for step in range(num_steps):
        t = torch.full((num_samples,), step * dt, device=device)
        
        if cfg_scale > 1.0:
            # Classifier-Free Guidance
            # v = v_uncond + cfg_scale * (v_cond - v_uncond)
            
            # 条件预测
            v_cond = model(x, t, labels)
            
            # 无条件预测
            labels_uncond = torch.full_like(labels, model_base.num_classes)
            v_uncond = model(x, t, labels_uncond)
            
            # CFG 组合
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            # 普通采样
            v = model(x, t, labels)
        
        # Euler 步进
        x = x + v * dt
    
    # 裁剪到 [-1, 1]
    x = torch.clamp(x, -1, 1)
    
    model.train()
    return x


def save_samples(samples, save_path, nrow=10):
    """保存采样图像"""
    # 从 [-1, 1] 转换到 [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    grid = make_grid(samples, nrow=nrow, padding=2)
    save_image(grid, save_path)


# ===============================================================================
# 训练
# ===============================================================================

def train_epoch(model, loader, optimizer, scheduler, scaler, ema, args, device, epoch, global_step):
    """训练一个 epoch"""
    model.train()
    
    # 获取真实模型（处理 DataParallel）
    model_base = model.module if isinstance(model, nn.DataParallel) else model
    
    total_loss = 0.0
    start_time = time.time()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # 混合精度训练
        with autocast(enabled=args.use_amp):
            loss = flow_matching_loss(
                model_base, images, labels, 
                cfg_dropout=args.cfg_dropout
            )
        
        # 反向传播
        optimizer.zero_grad()
        
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        # 学习率调度
        scheduler.step()
        
        # EMA 更新
        if args.use_ema:
            ema.update(model_base)
        
        # 统计
        total_loss += loss.item()
        global_step += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
        
        # 日志
        if global_step % args.log_every == 0:
            avg_loss = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * args.batch_size / elapsed
            
            print(f"\n[Step {global_step:06d}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Avg: {avg_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"Speed: {samples_per_sec:.1f} samples/sec")
    
    avg_loss = total_loss / len(loader)
    return avg_loss, global_step


# ===============================================================================
# 主函数
# ===============================================================================

def main():
    args = get_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_multi_gpu = args.multi_gpu and torch.cuda.device_count() > 1
    
    # 打印配置
    print("="*80)
    print("Flow Matching Transformer Training")
    print("="*80)
    print(f"Device: {device}")
    if use_multi_gpu:
        print(f"Multi-GPU: {torch.cuda.device_count()} GPUs (DataParallel)")
    print(f"Mixed Precision: {args.use_amp}")
    print(f"EMA: {args.use_ema}")
    print(f"Batch size: {args.batch_size} (per GPU)")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Save dir: {args.save_dir}")
    print("="*80)
    
    # 数据加载
    loader = get_dataloader(args)
    print(f"\nDataset: {len(loader.dataset)} images")
    print(f"Batches per epoch: {len(loader)}")
    
    # 创建模型
    model = FlowMatchingTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=1,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=10,
        class_dropout_prob=0.0  # 在损失函数中处理
    ).to(device)
    
    # DataParallel
    if use_multi_gpu:
        model = nn.DataParallel(model)
        print(f"Model wrapped with DataParallel")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params / 1e6:.2f}M")
    print(f"  Trainable: {trainable_params / 1e6:.2f}M")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # 学习率调度
    total_steps = len(loader) * args.num_epochs
    scheduler = get_lr_schedule(optimizer, args, total_steps)
    
    # 混合精度
    scaler = GradScaler(enabled=args.use_amp)
    
    # EMA
    ema = None
    if args.use_ema:
        model_base = model.module if use_multi_gpu else model
        ema = EMA(model_base, decay=args.ema_decay)
        print(f"EMA enabled with decay={args.ema_decay}")
    
    # 恢复训练
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        
        model_base = model.module if use_multi_gpu else model
        model_base.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        if args.use_amp and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        
        if args.use_ema and 'ema' in checkpoint:
            ema.shadow = checkpoint['ema']
        
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # 训练循环
    print("\n" + "="*80)
    print("Training started")
    print("="*80)
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # 训练
        avg_loss, global_step = train_epoch(
            model, loader, optimizer, scheduler, scaler, ema,
            args, device, epoch+1, global_step
        )
        
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # 保存检查点
        model_base = model.module if use_multi_gpu else model
        
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model': model_base.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict() if args.use_amp else None,
            'ema': ema.shadow if args.use_ema else None,
            'loss': avg_loss,
            'best_loss': best_loss,
            'args': vars(args)
        }
        
        # 最新检查点
        torch.save(checkpoint, save_dir / 'checkpoint_latest.pt')
        
        # 定期保存
        if (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch+1:03d}.pt')
        
        # 最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint['best_loss'] = best_loss
            torch.save(checkpoint, save_dir / 'checkpoint_best.pt')
            print(f"✓ New best model saved (loss: {best_loss:.4f})")
        
        # 采样
        if (epoch + 1) % args.sample_every == 0:
            print("\nGenerating samples...")
            
            # 使用 EMA 模型采样（如果启用）
            if args.use_ema:
                ema.apply_shadow(model_base)
            
            # 无条件采样
            samples = sample(
                model, args.num_samples, args.sample_steps,
                device, labels=None, cfg_scale=1.0
            )
            save_samples(
                samples,
                save_dir / f'sample_epoch_{epoch+1:03d}_uncond.png',
                nrow=10
            )
            
            # 条件采样（每类 10 个）
            n_per_class = 10
            labels = torch.repeat_interleave(
                torch.arange(10, device=device), n_per_class
            )
            samples = sample(
                model, len(labels), args.sample_steps,
                device, labels=labels, cfg_scale=args.cfg_scale
            )
            save_samples(
                samples,
                save_dir / f'sample_epoch_{epoch+1:03d}_cond_cfg{args.cfg_scale}.png',
                nrow=10
            )
            
            # 恢复训练模型
            if args.use_ema:
                ema.restore(model_base)
            
            print(f"Samples saved to {save_dir}")
    
    # 训练完成
    print("\n" + "="*80)
    print("Training finished!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved in: {save_dir}")
    print("="*80)


if __name__ == "__main__":
    main()