# inference.py
import torch
import torch.nn.functional as F
from config import T
from dit import DiT
import matplotlib.pyplot as plt
from diffusion import diffusion_schedule
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def backward_denoise(model, x, y=None, cfg_scale=1.0, use_cfg=False):
    """
    DiT去噪推理过程 (支持三种模式)
    
    Args:
        model: DiT模型
        x: [B, C, H, W] 初始噪声
        y: [B] 类别标签
            - None: 无条件生成
            - Tensor: 条件生成
        cfg_scale: CFG强度
            - 1.0: 纯条件/无条件生成
            - >1.0: 增强条件控制 (推荐 2.0-7.0)
        use_cfg: 是否使用CFG
            - False: 普通生成
            - True: CFG生成 (需要y不为None)
    
    Returns:
        steps: 去噪过程的中间步骤
    """
    steps = [x.clone().cpu()]
    
    x = x.to(DEVICE)
    if y is not None:
        y = y.to(DEVICE)
    
    # 将扩散调度器移到设备
    diffusion_schedule.to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        for time in range(T - 1, -1, -1):
            t = torch.full((x.size(0),), time, dtype=torch.long).to(DEVICE)
            
            # ===== 三种生成模式 =====
            if use_cfg and y is not None and cfg_scale != 1.0:
                # 模式1: CFG生成 (同时预测条件和无条件)
                noise_cond = model(x, t, y)           # 有标签
                noise_uncond = model(x, t, y=None)    # 无标签
                
                # CFG公式
                noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            
            elif y is not None:
                # 模式2: 纯条件生成
                noise = model(x, t, y)
            
            else:
                # 模式3: 纯无条件生成
                noise = model(x, t, y=None)
            
            # ===== 去噪步骤 =====
            shape = (x.size(0), 1, 1, 1)
            
            alpha_t = diffusion_schedule.alphas[t].view(*shape)
            alpha_cumprod_t = diffusion_schedule.alphas_cumprod[t].view(*shape)
            
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)) * noise
            )
            
            if time != 0:
                sigma_t = torch.sqrt(diffusion_schedule.variance[t].view(*shape))
                z = torch.randn_like(x)
                x = mean + sigma_t * z
            else:
                x = mean
            
            x = torch.clamp(x, -1.0, 1.0)
            
            # 保存关键步骤
            if time % (T // 20) == 0 or time == 0:
                steps.append(x.clone().cpu())
            
            if time % 100 == 0:
                print(f"去噪进度: {T - time}/{T}")
    
    return steps


# ==================== 使用示例 ====================

def load_model(checkpoint_path):
    """加载模型"""
    model = DiT(
        img_size=28,
        patch_size=4,
        in_channels=1,
        hidden_dim=768,
        depth=12,
        num_heads=12,
        num_classes=10,
        class_dropout_prob=0.1
    ).to(DEVICE)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def example_1_unconditional_generation(model, num_samples=10):
    """
    示例1: 无条件生成 (随机生成数字)
    """
    print("\n" + "=" * 60)
    print("示例1: 无条件生成")
    print("=" * 60)
    
    # 生成随机噪声
    x_T = torch.randn(num_samples, 1, 28, 28)
    
    # 无条件生成 (y=None)
    steps = backward_denoise(
        model=model,
        x=x_T,
        y=None,          # ✅ 关键: y=None表示无条件
        cfg_scale=1.0,   # 无条件时cfg_scale无效
        use_cfg=False
    )
    
    # 可视化
    visualize_results(steps[-1], title="Unconditional Generation")
    return steps


def example_2_conditional_generation(model, digits=None):
    """
    示例2: 条件生成 (生成指定数字)
    """
    print("\n" + "=" * 60)
    print("示例2: 条件生成")
    print("=" * 60)
    
    if digits is None:
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    num_samples = len(digits)
    x_T = torch.randn(num_samples, 1, 28, 28)
    y = torch.tensor(digits, dtype=torch.long)
    
    print(f"生成数字: {digits}")
    
    # 条件生成 (有标签,无CFG)
    steps = backward_denoise(
        model=model,
        x=x_T,
        y=y,             # ✅ 指定类别
        cfg_scale=1.0,   # 纯条件生成
        use_cfg=False
    )
    
    visualize_results(steps[-1], labels=digits, title="Conditional Generation")
    return steps


def example_3_cfg_generation(model, digits=None, cfg_scale=3.0):
    """
    示例3: CFG生成 (增强的条件生成)
    """
    print("\n" + "=" * 60)
    print(f"示例3: CFG生成 (scale={cfg_scale})")
    print("=" * 60)
    
    if digits is None:
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    num_samples = len(digits)
    x_T = torch.randn(num_samples, 1, 28, 28)
    y = torch.tensor(digits, dtype=torch.long)
    
    print(f"生成数字: {digits}")
    print(f"CFG强度: {cfg_scale}")
    
    # CFG生成
    steps = backward_denoise(
        model=model,
        x=x_T,
        y=y,
        cfg_scale=cfg_scale,  # ✅ CFG强度 (推荐2.0-7.0)
        use_cfg=True          # ✅ 启用CFG
    )
    
    visualize_results(
        steps[-1], 
        labels=digits, 
        title=f"CFG Generation (scale={cfg_scale})"
    )
    return steps


def compare_generation_modes(model, digit=3):
    """
    示例4: 对比三种生成模式
    """
    print("\n" + "=" * 60)
    print("示例4: 对比三种生成模式")
    print("=" * 60)
    
    # 使用相同的初始噪声
    x_T = torch.randn(1, 1, 28, 28)
    x_T_copy1 = x_T.clone()
    x_T_copy2 = x_T.clone()
    x_T_copy3 = x_T.clone()
    
    y = torch.tensor([digit])
    
    # 模式1: 无条件
    print("\n生成模式1: 无条件")
    steps_uncond = backward_denoise(model, x_T_copy1, y=None)
    
    # 模式2: 条件 (无CFG)
    print("\n生成模式2: 条件 (无CFG)")
    steps_cond = backward_denoise(model, x_T_copy2, y=y, cfg_scale=1.0)
    
    # 模式3: CFG
    print("\n生成模式3: CFG (scale=3.0)")
    steps_cfg = backward_denoise(model, x_T_copy3, y=y, cfg_scale=3.0, use_cfg=True)
    
    # 并排对比
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    images = [steps_uncond[-1], steps_cond[-1], steps_cfg[-1]]
    titles = ["Unconditional", f"Conditional (digit={digit})", f"CFG (scale=3.0, digit={digit})"]
    
    for ax, img, title in zip(axes, images, titles):
        img = (img[0] + 1.0) / 2.0
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('generation_modes_comparison.png', dpi=150)
    plt.show()


def compare_cfg_strengths(model, digit=7):
    """
    示例5: 对比不同CFG强度
    """
    print("\n" + "=" * 60)
    print("示例5: 对比不同CFG强度")
    print("=" * 60)
    
    cfg_scales = [1.0, 2.0, 3.0, 5.0, 7.0]
    y = torch.tensor([digit])
    
    fig, axes = plt.subplots(1, len(cfg_scales), figsize=(len(cfg_scales) * 2.5, 3))
    
    for i, scale in enumerate(cfg_scales):
        print(f"\nCFG scale = {scale}")
        x_T = torch.randn(1, 1, 28, 28)
        
        steps = backward_denoise(
            model, x_T, y=y, 
            cfg_scale=scale, 
            use_cfg=(scale > 1.0)
        )
        
        img = (steps[-1][0] + 1.0) / 2.0
        img = img.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'CFG={scale}', fontsize=12)
        axes[i].axis('off')
    
    plt.suptitle(f'CFG Strength Comparison (Target Digit: {digit})', fontsize=14)
    plt.tight_layout()
    plt.savefig('cfg_strength_comparison.png', dpi=150)
    plt.show()


def visualize_results(images, labels=None, title="Generated Images"):
    """可视化生成结果"""
    num_images = images.size(0)
    cols = min(10, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        row = i // cols
        col = i % cols
        
        img = (images[i] + 1.0) / 2.0
        img = img.permute(1, 2, 0).numpy()
        
        axes[row, col].imshow(img, cmap='gray')
        
        if labels is not None:
            axes[row, col].set_title(f'Label: {labels[i]}', fontsize=10)
        
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()


# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("DiT 推理 - 多种生成模式演示")
    print("=" * 60)
    
    # 加载模型
    model = load_model('/home/user/demo/review/Dit/ray_results/dit_training/checkpoint_2026-01-06_18-04-49.906488/checkpoint.pt')
    print("✅ 模型加载成功")
    
    # 示例1: 无条件生成
    example_1_unconditional_generation(model, num_samples=10)
    
    # 示例2: 条件生成
    example_2_conditional_generation(model, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    # 示例3: CFG生成
    example_3_cfg_generation(model, digits=[3, 3, 7, 7, 9, 9], cfg_scale=3.0)
    
    # 示例4: 对比三种模式
    compare_generation_modes(model, digit=5)
    
    # 示例5: 对比CFG强度
    compare_cfg_strengths(model, digit=8)
    
    print("\n" + "=" * 60)
    print("✅ 所有示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()