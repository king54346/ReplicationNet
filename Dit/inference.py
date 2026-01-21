import torch
import torch.nn.functional as F
from config import T
from dit import DiT
import matplotlib.pyplot as plt
from diffusion import *
from ray.train import Checkpoint
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def backward_denoise(model, x, y=None, cfg_scale=3.0, use_cfg=True):
    """
    DiT去噪推理过程

    Args:
        model: DiT模型
        x: [B, C, H, W] 初始噪声
        y: [B] 类别标签 (None表示无条件生成)
        cfg_scale: Classifier-Free Guidance强度 (>1.0增强条件控制)
        use_cfg: 是否使用CFG

    Returns:
        steps: 每个时间步的图像列表
    """
    steps = [x.clone()]

    # 全局变量移到设备
    global alphas, alphas_cumprod, variance
    x = x.to(DEVICE)
    alphas = alphas.to(DEVICE)
    alphas_cumprod = alphas_cumprod.to(DEVICE)
    variance = variance.to(DEVICE)

    if y is not None:
        y = y.to(DEVICE)

    model.eval()
    with torch.no_grad():
        # 从T-1到0逐步去噪
        for time in range(T - 1, -1, -1):
            t = torch.full((x.size(0),), time, dtype=torch.long).to(DEVICE)

            # ===== 1. 预测噪声 =====
            if use_cfg and y is not None:
                # Classifier-Free Guidance
                # 同时预测条件噪声和无条件噪声
                noise_cond = model(x, t, y)  # 条件预测
                noise_uncond = model(x, t, y=None)  # 无条件预测

                # CFG公式: noise = w * noise_cond + (1-w) * noise_uncond
                #         = noise_uncond + w * (noise_cond - noise_uncond)
                noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                # 普通预测
                noise = model(x, t, y)

            # ===== 2. 计算x_{t-1}的均值 =====
            # 根据DDPM公式: μ_θ(x_t, t) = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ)
            shape = (x.size(0), 1, 1, 1)

            alpha_t = alphas[t].view(*shape)  # α_t
            alpha_cumprod_t = alphas_cumprod[t].view(*shape)  # ᾱ_t

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                    x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)) * noise
            )

            # ===== 3. 添加随机噪声(除了最后一步) =====
            if time != 0:
                # σ_t = √β_t (或使用方差调度)
                sigma_t = torch.sqrt(variance[t].view(*shape))
                z = torch.randn_like(x)  # 标准正态噪声
                x = mean + sigma_t * z
            else:
                # t=0时不添加噪声
                x = mean

            # ===== 4. 裁剪到有效范围 =====
            x = torch.clamp(x, -1.0, 1.0)

            # 保存当前步骤
            if time % (T // 20) == 0 or time == 0:  # 每5%保存一次
                steps.append(x.clone().cpu())

            # 打印进度
            if time % 100 == 0:
                print(f"去噪进度: {T - time}/{T} (剩余{time}步)")

    return steps


def visualize_denoise_process(steps, save_path='denoise_process.png'):
    """
    可视化去噪过程

    Args:
        steps: backward_denoise返回的步骤列表
        save_path: 保存路径
    """
    batch_size = steps[0].size(0)
    num_steps = len(steps)

    fig, axes = plt.subplots(batch_size, num_steps, figsize=(num_steps * 2, batch_size * 2))

    for b in range(batch_size):
        for i, step_img in enumerate(steps):
            # 像素值从[-1,1]还原到[0,1]
            img = (step_img[b] + 1.0) / 2.0
            img = img.permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C]
            img = np.clip(img, 0, 1)

            ax = axes[b, i] if batch_size > 1 else axes[i]
            ax.imshow(img, cmap='gray' if img.shape[2] == 1 else None)
            ax.axis('off')

            # 第一行显示步骤号
            if b == 0:
                ax.set_title(f'Step {i}', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化结果已保存到: {save_path}")
    plt.show()


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("DiT 去噪推理过程")
    print("=" * 50)

    # ===== 1. 加载模型 =====
    print("\n📂 加载模型...")
    checkpoint = Checkpoint.from_directory(
        "/home/user/demo/review/Dit/ray_results/dit_training/checkpoint_2026-01-06_18-04-49.906488"
    )

    with checkpoint.as_directory() as checkpoint_dir:
        checkpoint_data = torch.load(f"{checkpoint_dir}/checkpoint.pt")
        model = DiT(
            img_size=28,
            patch_size=4,
            in_channels=1,  # 注意你的代码里用的是channel,应该统一
            hidden_dim=768,  # 对应你的emb_size
            depth=12,  # 对应你的dit_num
            num_heads=12,  # 对应你的head
            num_classes=10
        ).to(DEVICE)
        model.load_state_dict(checkpoint_data["model_state_dict"])

    print(f"✅ 模型已加载到 {DEVICE}")

    # ===== 2. 准备输入 =====
    batch_size = 10

    # 生成初始噪声 (从标准正态分布采样)
    x_T = torch.randn(batch_size, 1, 28, 28)

    # 生成类别标签 (0-9各一个)
    y = torch.arange(0, 10, dtype=torch.long)

    print(f"\n🎲 生成 {batch_size} 个初始噪声")
    print(f"📝 类别标签: {y.tolist()}")

    # ===== 3. 执行去噪 =====
    print("\n🔄 开始去噪推理...")
    print(f"总步数: {T}")
    print(f"CFG强度: 3.0")

    steps = backward_denoise(
        model=model,
        x=x_T,
        y=y,
        cfg_scale=7.0,  # CFG强度 (1.0=无CFG, 7.0=强条件控制)
        use_cfg=True  # 启用CFG
    )

    print(f"\n✅ 去噪完成! 共保存 {len(steps)} 个中间步骤")

    # ===== 4. 可视化结果 =====
    print("\n📊 生成可视化...")
    visualize_denoise_process(steps, save_path='denoise_process.png')

    # ===== 5. 保存最终结果 =====
    final_images = steps[-1]  # [B, C, H, W]

    # 单独保存每个数字
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        img = (final_images[i] + 1.0) / 2.0
        img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {i}', fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('final_results.png', dpi=150, bbox_inches='tight')
    print("✅ 最终结果已保存到: final_results.png")
    plt.show()

    print("\n" + "=" * 50)
    print("推理完成!")
    print("=" * 50)


# ==================== 额外功能 ====================

def generate_specific_digit(model, digit, num_samples=5, cfg_scale=3.0):
    """
    生成指定数字的多个样本

    Args:
        model: DiT模型
        digit: 要生成的数字 (0-9)
        num_samples: 生成数量
        cfg_scale: CFG强度
    """
    x_T = torch.randn(num_samples, 1, 28, 28)
    y = torch.full((num_samples,), digit, dtype=torch.long)

    print(f"\n生成 {num_samples} 个数字 '{digit}' 的样本...")
    steps = backward_denoise(model, x_T, y, cfg_scale=cfg_scale)

    # 可视化
    final = steps[-1]
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    for i in range(num_samples):
        img = (final[i] + 1.0) / 2.0
        img = img.permute(1, 2, 0).numpy()
        ax = axes[i] if num_samples > 1 else axes
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.suptitle(f'Generated Digit: {digit}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'digit_{digit}_samples.png', dpi=150)
    plt.show()


def compare_cfg_strengths(model, digit=3):
    """
    比较不同CFG强度的效果
    """
    cfg_scales = [1.0, 2.0, 3.0, 5.0, 7.0]

    fig, axes = plt.subplots(1, len(cfg_scales), figsize=(len(cfg_scales) * 2, 2))

    for i, scale in enumerate(cfg_scales):
        x_T = torch.randn(1, 1, 28, 28)
        y = torch.tensor([digit])

        steps = backward_denoise(model, x_T, y, cfg_scale=scale)
        img = (steps[-1][0] + 1.0) / 2.0
        img = img.permute(1, 2, 0).numpy()

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'CFG={scale}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle(f'CFG Scale Comparison (Digit {digit})', fontsize=14)
    plt.tight_layout()
    plt.savefig('cfg_comparison.png', dpi=150)
    plt.show()