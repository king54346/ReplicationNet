"""
Flow Matching Transformer - 采样脚本
从训练好的模型生成图像
"""

import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from pathlib import Path
import argparse
from tqdm import tqdm

from flowmatchingTransformer import FlowMatchingTransformer


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
    
    # ODE 求解（带进度条）
    dt = 1.0 / num_steps
    
    for step in tqdm(range(num_steps), desc='Sampling'):
        t = torch.full((num_samples,), step * dt, device=device)
        
        if cfg_scale > 1.0:
            # Classifier-Free Guidance
            v_cond = model(x, t, labels)
            labels_uncond = torch.full_like(labels, model_base.num_classes)
            v_uncond = model(x, t, labels_uncond)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x, t, labels)
        
        # Euler 步进
        x = x + v * dt
    
    # 裁剪到 [-1, 1]
    x = torch.clamp(x, -1, 1)
    
    return x


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取配置
    args = checkpoint.get('args', {})
    
    # 创建模型
    model = FlowMatchingTransformer(
        img_size=args.get('image_size', 32),
        patch_size=args.get('patch_size', 4),
        in_channels=1,
        hidden_dim=args.get('hidden_dim', 384),
        depth=args.get('depth', 12),
        num_heads=args.get('num_heads', 6),
        mlp_ratio=args.get('mlp_ratio', 4.0),
        num_classes=10,
        class_dropout_prob=0.0
    ).to(device)
    
    # 加载权重（优先使用 EMA）
    if 'ema' in checkpoint and checkpoint['ema'] is not None:
        print("Using EMA weights")
        # 手动加载 EMA 权重
        for name, param in model.named_parameters():
            if name in checkpoint['ema']:
                param.data = checkpoint['ema'][name]
    else:
        print("Using regular weights")
        model.load_state_dict(checkpoint['model'])
    
    model.eval()
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully!")
    print(f"  Parameters: {total_params / 1e6:.2f}M")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Loss: {checkpoint.get('loss', 'unknown'):.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained model')
    
    # 模型
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    
    # 采样
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--num-steps', type=int, default=100,
                        help='Number of sampling steps')
    parser.add_argument('--cfg-scale', type=float, default=2.0,
                        help='Classifier-Free Guidance scale (1.0 = disabled)')
    
    # 条件
    parser.add_argument('--class-id', type=int, default=None,
                        help='Generate specific class (0-9, None = all classes)')
    parser.add_argument('--unconditional', action='store_true',
                        help='Unconditional generation')
    
    # 保存
    parser.add_argument('--output-dir', type=str, default='./samples',
                        help='Output directory')
    parser.add_argument('--nrow', type=int, default=10,
                        help='Number of images per row in grid')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 准备标签
    if args.unconditional:
        print(f"\nGenerating {args.num_samples} unconditional samples...")
        labels = None
        cfg_scale = 1.0  # 无条件时禁用 CFG
    elif args.class_id is not None:
        print(f"\nGenerating {args.num_samples} samples of class {args.class_id}...")
        labels = torch.full((args.num_samples,), args.class_id, device=device)
        cfg_scale = args.cfg_scale
    else:
        print(f"\nGenerating {args.num_samples} samples (all classes)...")
        # 每类生成相同数量
        n_per_class = args.num_samples // 10
        labels = torch.repeat_interleave(torch.arange(10, device=device), n_per_class)
        # 补齐到 num_samples
        if len(labels) < args.num_samples:
            extra = args.num_samples - len(labels)
            labels = torch.cat([labels, torch.arange(extra, device=device)])
        cfg_scale = args.cfg_scale
    
    print(f"Sampling steps: {args.num_steps}")
    print(f"CFG scale: {cfg_scale}")
    
    # 生成样本
    samples = sample(
        model, args.num_samples, args.num_steps,
        device, labels, cfg_scale
    )
    
    # 转换到 [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # 保存单独的图像
    for i, img in enumerate(samples):
        if labels is not None:
            label = labels[i].item()
            filename = f'sample_{i:04d}_class_{label}.png'
        else:
            filename = f'sample_{i:04d}.png'
        
        save_image(img, output_dir / filename)
    
    print(f"\nSaved {len(samples)} individual images to {output_dir}")
    
    # 保存网格
    grid = make_grid(samples, nrow=args.nrow, padding=2)
    
    if args.unconditional:
        grid_filename = f'grid_uncond_cfg{cfg_scale:.1f}.png'
    elif args.class_id is not None:
        grid_filename = f'grid_class{args.class_id}_cfg{cfg_scale:.1f}.png'
    else:
        grid_filename = f'grid_all_classes_cfg{cfg_scale:.1f}.png'
    
    save_image(grid, output_dir / grid_filename)
    print(f"Saved grid to {output_dir / grid_filename}")
    
    # 打印统计信息
    print("\nSample statistics:")
    print(f"  Shape: {samples.shape}")
    print(f"  Mean: {samples.mean():.4f}")
    print(f"  Std: {samples.std():.4f}")
    print(f"  Min: {samples.min():.4f}")
    print(f"  Max: {samples.max():.4f}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()