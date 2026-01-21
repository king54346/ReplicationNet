"""
Flow Matching 推理脚本 - 简化版
生成 MNIST 手写数字图像
"""

import os
import math
import torch
from torchvision import utils
from tqdm import tqdm
import argparse

from flowmatching import FlowMatchingModel


@torch.no_grad()
def sample(model, n, steps, device, labels=None, cfg_scale=0.0):
    """
    Flow Matching 采样
    
    Args:
        model: 模型
        n: 样本数量
        steps: 积分步数
        device: 设备
        labels: 类别标签 [n] (可选)
        cfg_scale: CFG 强度 (0 = 无条件)
    
    Returns:
        生成的图像 [n,1,28,28] in [-1,1]
    """
    model.eval()
    
    # 从噪声开始
    x = torch.randn(n, 1, 28, 28, device=device)
    dt = 1.0 / steps
    
    for i in tqdm(range(steps), desc="Sampling"):
        t = torch.full((n,), i / steps, device=device, dtype=torch.float32)
        
        # Classifier-Free Guidance
        if cfg_scale > 0 and labels is not None:
            v_cond = model(x, t, labels)
            v_uncond = model(x, t, None)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x, t, labels)
        
        # Euler 积分
        x = x + v * dt
    
    return x.clamp(-1, 1)


def save_grid(images, path, nrow=None):
    """保存图像网格"""
    out = (images + 1.0) / 2.0  # [-1,1] -> [0,1]
    if nrow is None:
        nrow = int(math.sqrt(len(images)) + 0.999)
    grid = utils.make_grid(out, nrow=nrow, padding=2)
    utils.save_image(grid, path)
    print(f"✓ Saved: {path}")


def load_model(checkpoint_path, device):
    """加载模型"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    model = FlowMatchingModel(
        img_channels=1,
        base_channels=64,
        channel_mult=(1, 2, 4),
        time_dim=256,
        num_classes=10,
        num_res_blocks=2,
        dropout=0.0,  # 推理不用 dropout
        use_attention=(False, True, True),
        num_heads=4
    ).to(device)
    
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    print(f"✓ Model loaded (epoch {ckpt.get('epoch', '?')})")
    return model


def main():
    parser = argparse.ArgumentParser(description="Flow Matching Inference")
    
    # 必需参数
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="模型检查点路径")
    
    # 生成模式
    parser.add_argument("--mode", type=str, default="conditional",
                       choices=["unconditional", "conditional", "digits"],
                       help="生成模式")
    parser.add_argument("--digits", type=int, nargs="+", default=None,
                       help="指定生成的数字，如: --digits 0 1 2 3")
    
    # 采样参数
    parser.add_argument("--num_samples", type=int, default=64,
                       help="生成样本数（无条件模式）")
    parser.add_argument("--samples_per_class", type=int, default=8,
                       help="每类样本数（条件模式）")
    parser.add_argument("--steps", type=int, default=100,
                       help="采样步数")
    parser.add_argument("--cfg_scale", type=float, default=2.0,
                       help="CFG 强度 (0=无条件)")
    
    # 输出
    parser.add_argument("--output_dir", type=str, default="./outputs/inference",
                       help="输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Flow Matching Inference")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Steps: {args.steps}")
    print(f"CFG Scale: {args.cfg_scale}")
    print("=" * 80)
    
    # 加载模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    
    # === 模式 1: 无条件生成 ===
    if args.mode == "unconditional":
        print(f"\n生成 {args.num_samples} 个无条件样本...")
        
        samples = sample(model, args.num_samples, args.steps, device)
        save_grid(samples, os.path.join(args.output_dir, "unconditional.png"))
    
    # === 模式 2: 条件生成（所有类别）===
    elif args.mode == "conditional":
        print(f"\n生成条件样本 (每类 {args.samples_per_class} 个)...")
        
        # 0-9 每个数字
        labels = torch.repeat_interleave(
            torch.arange(10, device=device),
            args.samples_per_class
        )
        
        samples = sample(
            model, 
            len(labels), 
            args.steps, 
            device,
            labels=labels,
            cfg_scale=args.cfg_scale
        )
        
        save_grid(
            samples,
            os.path.join(args.output_dir, f"conditional_cfg{args.cfg_scale}.png"),
            nrow=args.samples_per_class
        )
    
    # === 模式 3: 指定数字 ===
    elif args.mode == "digits":
        if args.digits is None:
            args.digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        print(f"\n生成指定数字: {args.digits} (每个 {args.samples_per_class} 个)...")
        
        labels = torch.repeat_interleave(
            torch.tensor(args.digits, device=device),
            args.samples_per_class
        )
        
        samples = sample(
            model,
            len(labels),
            args.steps,
            device,
            labels=labels,
            cfg_scale=args.cfg_scale
        )
        
        save_grid(
            samples,
            os.path.join(args.output_dir, f"digits_{''.join(map(str, args.digits))}.png"),
            nrow=args.samples_per_class
        )
    
    print("\n" + "=" * 80)
    print("✓ 完成!")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)

# python inference.py --checkpoint checkpoint_best.pt --steps 200 --cfg_scale 3.0
if __name__ == "__main__":
    main()