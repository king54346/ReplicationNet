"""
Flow Matching Training - 简化版
支持单GPU和多GPU (DataParallel) 训练
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
import argparse

from flowmatching import FlowMatchingModel


# ===============================================================================
# 配置
# ===============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Flow Matching Training")
    
    # 数据
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=32)
    
    # 模型
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # 训练
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # CFG
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    
    # 采样
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=64)
    
    # 保存
    parser.add_argument("--save_dir", type=str, default="./outputs/flow_matching")
    parser.add_argument("--log_every", type=int, default=100)
    
    # 设备
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--multi_gpu", action="store_true", help="使用所有GPU (DataParallel)")
    
    # 恢复
    parser.add_argument("--resume", type=str, default=None)
    
    return parser.parse_args()


# ===============================================================================
# 数据
# ===============================================================================

def get_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
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
# 采样
# ===============================================================================

@torch.no_grad()
def sample(model, n, steps, device, labels=None, cfg_scale=0.0):
    """快速采样"""
    model_eval = model.module if isinstance(model, nn.DataParallel) else model
    was_training = model_eval.training
    model_eval.eval()
    
    x = torch.randn(n, 1, 32, 32, device=device)
    dt = 1.0 / steps
    
    for i in range(steps):
        t = torch.full((n,), i / steps, device=device, dtype=torch.float32)
        
        if cfg_scale > 0 and labels is not None:
            v_c = model(x, t, labels)
            v_u = model(x, t, None)
            v = v_u + cfg_scale * (v_c - v_u)
        else:
            v = model(x, t, labels)
        
        x = x + v * dt
    
    if was_training:
        model_eval.train()
    
    return x.clamp(-1, 1)


def save_grid(images, path):
    """保存图像网格"""
    out = (images.clamp(-1, 1) + 1.0) / 2.0
    nrow = int(math.sqrt(len(images)) + 0.999)
    grid = utils.make_grid(out, nrow=nrow, padding=2)
    utils.save_image(grid, path)


# ===============================================================================
# 训练
# ===============================================================================

def train_epoch(model, loader, optimizer, epoch, args, device, step=0):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    count = 0
    
    # 获取真实模型（处理 DataParallel）
    model_base = model.module if isinstance(model, nn.DataParallel) else model
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        B = images.shape[0]
        
        # 归一化到 [-1, 1]
        z = images * 2.0 - 1.0
        
        # CFG: 随机丢弃标签
        mask = torch.rand(B, device=device) < args.cfg_dropout
        labels_train = labels.clone()
        labels_train[mask] = model_base.label_embedding.num_embeddings - 1
        
        # Flow Matching
        x_0 = torch.randn_like(z)
        t = torch.rand(B, device=device, dtype=torch.float32)
        t_exp = t.view(B, 1, 1, 1)
        x_t = t_exp * z + (1 - t_exp) * x_0
        v_target = (z - x_0).detach()
        
        # 前向 + 反向
        optimizer.zero_grad()
        v_pred = model(x_t, t, labels_train)
        loss = nn.functional.mse_loss(v_pred, v_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        count += 1
        step += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        if step % args.log_every == 0:
            print(f"\n[Step {step:06d}] Loss: {loss.item():.6f}")
    
    return total_loss / count, step


def main():
    args = get_args()
    
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_multi_gpu = args.multi_gpu and torch.cuda.device_count() > 1
    
    print("="*80)
    print("Flow Matching Training")
    print("="*80)
    print(f"Device: {device}")
    if use_multi_gpu:
        print(f"Multi-GPU: {torch.cuda.device_count()} GPUs")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Save dir: {args.save_dir}")
    print("="*80)
    
    # 数据
    loader = get_dataloader(args)
    print(f"\nDataset: {len(loader.dataset)} images")
    print(f"Batches: {len(loader)}")
    
    # 模型
    model = FlowMatchingModel(
        img_channels=1,
        base_channels=args.base_channels,
        channel_mult=(1, 2, 4),
        time_dim=256,
        num_classes=10,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        use_attention=(False, True, True),
        num_heads=4
    ).to(device)
    
    if use_multi_gpu:
        model = nn.DataParallel(model)
        print(f"Model wrapped with DataParallel")
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # 恢复
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        
        if use_multi_gpu:
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])
        
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("step", 0)
        print(f"Resumed from epoch {start_epoch}")
    
    # 训练
    print("\n" + "="*80)
    print("Training started")
    print("="*80)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        
        # 训练
        avg_loss, global_step = train_epoch(
            model, loader, optimizer, epoch, args, device, global_step
        )
        print(f"Average Loss: {avg_loss:.6f}")
        
        # 保存
        model_to_save = model.module if use_multi_gpu else model
        
        ckpt = {
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "loss": avg_loss,
        }
        
        # 最新
        torch.save(ckpt, os.path.join(args.save_dir, "checkpoint_latest.pt"))
        
        # 每个epoch
        if epoch % 5 == 0:
            torch.save(ckpt, os.path.join(args.save_dir, f"checkpoint_epoch_{epoch:03d}.pt"))
        
        # 最佳
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, os.path.join(args.save_dir, "checkpoint_best.pt"))
            print(f"✓ Best model saved (loss: {best_loss:.6f})")
        
        # 采样
        if epoch % args.sample_every == 0:
            print("Generating samples...")
            
            # 无条件
            imgs = sample(model, args.num_samples, args.sample_steps, device)
            save_grid(imgs, os.path.join(args.save_dir, f"sample_epoch_{epoch:03d}.png"))
            
            # 条件
            n_per_class = args.num_samples // 10
            labels = torch.repeat_interleave(torch.arange(10, device=device), n_per_class)
            imgs = sample(model, len(labels), args.sample_steps, device, labels, args.cfg_scale)
            save_grid(imgs, os.path.join(args.save_dir, f"sample_epoch_{epoch:03d}_cond.png"))
            
            print(f"Saved samples to {args.save_dir}")
    
    print("\n" + "="*80)
    print("Training finished!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Models saved in: {args.save_dir}")
    print("="*80)

# python train.py --multi_gpu --resume ./outputs/flow_matching/checkpoint_latest.pt
if __name__ == "__main__":
    main()