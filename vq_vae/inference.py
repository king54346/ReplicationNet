"""
VAE推理脚本
支持图像重构、随机生成、潜在空间插值、算术运算等功能

主要改进:
1. 兼容所有三种VAE架构 (VAE/ConvVAE/DeepConvVAE)
2. 智能模型加载和配置推断
3. 更多可视化选项
4. 潜在空间探索工具
5. 批量处理支持
6. 更好的错误处理
"""
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision import transforms
from tqdm import tqdm

from model import VAE, ConvVAE, DeepConvVAE


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = 'cuda'
) -> Tuple[torch.nn.Module, dict, str]:
    """
    智能加载模型 - 自动检测模型类型和配置
    
    加载策略:
    1. 优先使用 config.json (如果存在)
    2. 从checkpoint中读取 model_type (如果存在)
    3. 从 state_dict 推断模型架构
    4. 使用默认配置作为后备
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
        
    Returns:
        model: 加载好的VAE模型
        config: 模型配置字典
        model_type: 模型类型字符串
    """
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / 'config.json'
    
    print(f"Loading model from: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # 方法1: 从配置文件加载
    if config_path.exists():
        print(f"✓ Found config file: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model_type = config.get('model_type', 'vae')
        
        # 从checkpoint验证latent_dim（config.json可能不准确）
        if 'fc_mu.weight' in state_dict:
            actual_latent_dim = state_dict['fc_mu.weight'].shape[0]
            config['latent_dim'] = actual_latent_dim
        
    # 方法2: 从checkpoint读取
    elif 'model_type' in checkpoint:
        print(f"✓ Found model_type in checkpoint")
        model_type = checkpoint['model_type']
        config = {
            'latent_dim': checkpoint.get('latent_dim', 20),
            'model_type': model_type
        }
        
    # 方法3: 从state_dict推断
    else:
        print(f"⚠ Config not found, inferring from state_dict...")
        model_type, config = infer_model_config(state_dict)
    
    # 创建模型
    model = create_model_from_config(model_type, config, device)
    
    # 加载权重
    try:
        model.load_state_dict(state_dict)
        print(f"✓ Model weights loaded successfully")
    except RuntimeError as e:
        print(f"⚠ Model architecture mismatch!")
        print(f"  Error: {e}")
        print(f"  Trying to load with strict=False...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print(f"✓ Loaded with some missing/unexpected keys")
    
    model.eval()
    
    print(f"\nModel Info:")
    print(f"  Type: {model_type}")
    print(f"  Latent dim: {config.get('latent_dim', 'unknown')}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config, model_type


def infer_model_config(state_dict: dict) -> Tuple[str, dict]:
    """从state_dict推断模型类型和配置"""
    # 检查DeepConvVAE特征
    has_res_blocks = any('encoder_res' in k for k in state_dict.keys())
    has_fc_decode = 'fc_decode.weight' in state_dict
    
    # 检查是否是卷积层
    first_layer_key = 'encoder.0.weight'
    is_conv = first_layer_key in state_dict and len(state_dict[first_layer_key].shape) == 4
    
    latent_dim = state_dict['fc_mu.weight'].shape[0]
    
    if has_res_blocks and has_fc_decode:
        # DeepConvVAE
        model_type = 'deep_conv_vae'
        print("  Detected: DeepConvVAE")
        
        # 推断通道数
        channels = []
        for key in sorted(state_dict.keys()):
            if key.startswith('encoder.') and '.weight' in key:
                w = state_dict[key]
                if len(w.shape) == 4:
                    channels.append(w.shape[0])
        hidden_channels = sorted(set(channels))[:3] if channels else [32, 64, 128]
        
        config = {
            'model_type': model_type,
            'latent_dim': latent_dim,
            'hidden_channels': hidden_channels,
            'in_channels': 1,
            'image_size': 28,
            'num_res_blocks': 2
        }
        
    elif is_conv:
        # ConvVAE
        model_type = 'conv_vae'
        print("  Detected: ConvVAE")
        
        channels = []
        for key in sorted(state_dict.keys()):
            if key.startswith('encoder.') and '.weight' in key:
                w = state_dict[key]
                if len(w.shape) == 4:
                    channels.append(w.shape[0])
        hidden_channels = sorted(set(channels))[:2] if channels else [32, 64]
        
        config = {
            'model_type': model_type,
            'latent_dim': latent_dim,
            'hidden_channels': hidden_channels,
            'in_channels': 1,
            'image_size': 28
        }
        
    else:
        # VAE (全连接)
        model_type = 'vae'
        print("  Detected: VAE (fully connected)")
        
        dims = []
        for key in sorted(state_dict.keys()):
            if key.startswith('encoder.') and '.weight' in key:
                w = state_dict[key]
                if len(w.shape) == 2:
                    dims.append(w.shape[0])
        hidden_dims = sorted(set(dims), reverse=True)[:2] if dims else [512, 256]
        
        config = {
            'model_type': model_type,
            'latent_dim': latent_dim,
            'hidden_dims': hidden_dims,
            'input_dim': 784
        }
    
    return model_type, config


def create_model_from_config(
    model_type: str,
    config: dict,
    device: str
) -> torch.nn.Module:
    """根据配置创建模型"""
    if model_type == 'vae':
        model = VAE(
            input_dim=config.get('input_dim', 784),
            hidden_dims=config.get('hidden_dims', [512, 256]),
            latent_dim=config.get('latent_dim', 20)
        )
    elif model_type == 'conv_vae':
        model = ConvVAE(
            in_channels=config.get('in_channels', 1),
            image_size=config.get('image_size', 28),
            hidden_channels=config.get('hidden_channels', [32, 64]),
            latent_dim=config.get('latent_dim', 20)
        )
    elif model_type == 'deep_conv_vae':
        model = DeepConvVAE(
            in_channels=config.get('in_channels', 1),
            image_size=config.get('image_size', 28),
            hidden_channels=config.get('hidden_channels', [32, 64, 128]),
            latent_dim=config.get('latent_dim', 32),
            num_res_blocks=config.get('num_res_blocks', 2)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


class VAEInference:
    """VAE推理器 - 支持所有VAE架构"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda'
    ):
        """
        Args:
            checkpoint_path: 模型检查点路径
            device: 设备
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 智能加载模型
        self.model, self.config, self.model_type = load_model_from_checkpoint(
            checkpoint_path,
            self.device
        )
        
        self.latent_dim = self.config['latent_dim']
        
        print(f"✓ Inference engine ready on {self.device}")
    
    @torch.no_grad()
    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """
        重构图像
        
        Args:
            images: 输入图像 [B, C, H, W]
            
        Returns:
            重构的图像 [B, C, H, W]
        """
        images = images.to(self.device)
        recon_images, _, _ = self.model(images)
        
        # VAE返回flatten的输出，需要reshape
        if self.model_type == 'vae':
            recon_images = recon_images.view(-1, 1, 28, 28)
        
        return recon_images
    
    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码到潜在空间（返回均值，不带随机性）
        
        Args:
            images: 输入图像 [B, C, H, W]
            
        Returns:
            潜在向量 [B, latent_dim]
        """
        images = images.to(self.device)
        
        # VAE需要flatten输入
        if self.model_type == 'vae':
            batch_size = images.size(0)
            images = images.view(batch_size, -1)
        
        mu, _ = self.model.encode(images)
        return mu
    
    @torch.no_grad()
    def decode(self, latent_vectors: torch.Tensor) -> torch.Tensor:
        """
        从潜在空间解码
        
        Args:
            latent_vectors: 潜在向量 [B, latent_dim]
            
        Returns:
            解码的图像 [B, C, H, W]
        """
        latent_vectors = latent_vectors.to(self.device)
        images = self.model.decode(latent_vectors)
        
        # ConvVAE和DeepConvVAE已经返回正确形状
        # VAE需要reshape
        if self.model_type == 'vae':
            images = images.view(-1, 1, 28, 28)
        
        return images
    
    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 16,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        随机生成新样本
        
        Args:
            num_samples: 生成样本数量
            temperature: 采样温度（控制多样性）
                - temperature=1.0: 标准正态分布
                - temperature>1.0: 更多样化
                - temperature<1.0: 更确定性
        
        Returns:
            生成的图像 [num_samples, C, H, W]
        """
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        z = z * temperature
        
        images = self.decode(z)
        return images
    
    @torch.no_grad()
    def interpolate(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        num_steps: int = 10,
        mode: str = 'linear'
    ) -> torch.Tensor:
        """
        在两个图像之间进行潜在空间插值
        
        Args:
            image1: 起始图像 [C, H, W] 或 [1, C, H, W]
            image2: 终止图像 [C, H, W] 或 [1, C, H, W]
            num_steps: 插值步数
            mode: 插值模式
                - 'linear': 线性插值（快速）
                - 'spherical': 球面插值（更平滑，保持范数）
        
        Returns:
            插值序列 [num_steps, C, H, W]
        """
        # 确保输入是4D [1, C, H, W]
        if image1.dim() == 3:
            image1 = image1.unsqueeze(0)
        if image2.dim() == 3:
            image2 = image2.unsqueeze(0)
        
        # 编码到潜在空间
        z1 = self.encode(image1).squeeze(0)
        z2 = self.encode(image2).squeeze(0)
        
        # 生成插值系数
        alphas = torch.linspace(0, 1, num_steps).to(self.device)
        
        interpolations = []
        for alpha in alphas:
            if mode == 'linear':
                # 线性插值: z = (1-α)z₁ + αz₂
                z_interp = (1 - alpha) * z1 + alpha * z2
                
            elif mode == 'spherical':
                # 球面线性插值 (SLERP)
                # 保持向量范数，沿大圆弧插值
                
                # 计算夹角
                dot_product = torch.dot(z1, z2)
                norm_product = torch.norm(z1) * torch.norm(z2)
                cos_omega = torch.clamp(dot_product / norm_product, -1, 1)
                omega = torch.acos(cos_omega)
                
                # 如果向量几乎平行，回退到线性插值
                if omega.abs() < 1e-6:
                    z_interp = (1 - alpha) * z1 + alpha * z2
                else:
                    # SLERP公式: z = [sin((1-α)Ω)/sin(Ω)]z₁ + [sin(αΩ)/sin(Ω)]z₂
                    sin_omega = torch.sin(omega)
                    z_interp = (torch.sin((1 - alpha) * omega) / sin_omega) * z1 + \
                               (torch.sin(alpha * omega) / sin_omega) * z2
            else:
                raise ValueError(f"Unknown interpolation mode: {mode}")
            
            # 解码
            img = self.decode(z_interp.unsqueeze(0))
            interpolations.append(img.squeeze(0))
        
        return torch.stack(interpolations)
    
    @torch.no_grad()
    def latent_arithmetic(
        self,
        images: List[torch.Tensor],
        operations: List[str]
    ) -> torch.Tensor:
        """
        潜在空间算术运算
        
        示例: 生成 "数字风格迁移"
        - images[0]: 源数字 (如 "3")
        - images[1]: 风格参考 (如粗体的 "5")
        - images[2]: 风格参考对照 (如正常的 "5")
        - 操作: ['+', '-'] → z₀ + z₁ - z₂
        
        Args:
            images: 图像列表 [N个 (C, H, W)]
            operations: 操作符列表 [N-1个 '+' 或 '-']
            
        Returns:
            结果图像 [C, H, W]
        """
        # 确保所有图像都是4D
        images = [img.unsqueeze(0) if img.dim() == 3 else img for img in images]
        
        # 编码所有图像
        latents = [self.encode(img).squeeze(0) for img in images]
        
        # 执行算术运算
        result = latents[0]
        for i, op in enumerate(operations):
            if op == '+':
                result = result + latents[i + 1]
            elif op == '-':
                result = result - latents[i + 1]
            elif op == '*':
                # 标量乘法
                result = result * latents[i + 1]
            else:
                raise ValueError(f"Unknown operation: {op}")
        
        # 解码结果
        return self.decode(result.unsqueeze(0)).squeeze(0)
    
    @torch.no_grad()
    def explore_latent_dimensions(
        self,
        base_image: torch.Tensor,
        dim_indices: List[int],
        value_range: Tuple[float, float] = (-3, 3),
        num_steps: int = 7
    ) -> torch.Tensor:
        """
        探索特定潜在维度的影响
        
        固定其他维度，只改变指定维度的值，观察生成结果
        
        Args:
            base_image: 基准图像 [C, H, W]
            dim_indices: 要探索的维度索引列表
            value_range: 值域范围 (min, max)
            num_steps: 每个维度的采样步数
            
        Returns:
            结果网格 [len(dim_indices) * num_steps, C, H, W]
        """
        if base_image.dim() == 3:
            base_image = base_image.unsqueeze(0)
        
        # 获取基准潜在向量
        z_base = self.encode(base_image).squeeze(0)
        
        # 生成采样值
        values = torch.linspace(value_range[0], value_range[1], num_steps)
        
        results = []
        for dim_idx in dim_indices:
            for value in values:
                z_modified = z_base.clone()
                z_modified[dim_idx] = value
                
                img = self.decode(z_modified.unsqueeze(0))
                results.append(img.squeeze(0))
        
        return torch.stack(results)
    
    @torch.no_grad()
    def visualize_latent_space_2d(
        self,
        test_loader,
        num_samples: int = 1000,
        method: str = 'pca',
        save_path: Optional[str] = None
    ):
        """
        可视化潜在空间（降维到2D）
        
        Args:
            test_loader: 测试数据加载器
            num_samples: 使用的样本数
            method: 降维方法 ('pca', 'tsne', 'umap')
            save_path: 保存路径
        """
        latents = []
        labels = []
        
        print(f"Collecting {num_samples} samples for visualization...")
        for i, (images, lbls) in enumerate(tqdm(test_loader)):
            if len(latents) * test_loader.batch_size >= num_samples:
                break
            
            z = self.encode(images)
            latents.append(z.cpu().numpy())
            labels.append(lbls.numpy())
        
        latents = np.concatenate(latents, axis=0)[:num_samples]
        labels = np.concatenate(labels, axis=0)[:num_samples]
        
        print(f"Performing {method.upper()} dimensionality reduction...")
        
        # 降维
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            latents_2d = reducer.fit_transform(latents)
            explained_var = reducer.explained_variance_ratio_
            title = f'VAE Latent Space (PCA, {explained_var[0]:.1%}+{explained_var[1]:.1%} var)'
            
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            latents_2d = reducer.fit_transform(latents)
            title = 'VAE Latent Space (t-SNE)'
            
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                latents_2d = reducer.fit_transform(latents)
                title = 'VAE Latent Space (UMAP)'
            except ImportError:
                print("⚠ UMAP not installed, falling back to PCA")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                latents_2d = reducer.fit_transform(latents)
                title = 'VAE Latent Space (PCA)'
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 可视化
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            latents_2d[:, 0],
            latents_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
        
        cbar = plt.colorbar(scatter, label='Digit', ticks=range(10))
        cbar.set_label('Digit', fontsize=12)
        
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Latent space visualization saved to {save_path}")
        
        plt.show()
    
    @torch.no_grad()
    def get_reconstruction_error(
        self,
        test_loader,
        num_batches: int = 10
    ) -> Tuple[float, np.ndarray]:
        """
        计算重构误差统计
        
        Returns:
            mean_error: 平均重构误差
            per_sample_errors: 每个样本的误差 [N]
        """
        errors = []
        
        print("Computing reconstruction errors...")
        for i, (images, _) in enumerate(tqdm(test_loader)):
            if i >= num_batches:
                break
            
            images = images.to(self.device)
            recon = self.reconstruct(images)
            
            # 计算每个样本的MSE
            batch_errors = torch.mean(
                (images - recon) ** 2,
                dim=(1, 2, 3)
            ).cpu().numpy()
            
            errors.extend(batch_errors)
        
        errors = np.array(errors)
        return errors.mean(), errors


def visualize_reconstruction(
    inference: VAEInference,
    images: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = 'Image Reconstruction'
):
    """可视化重构结果"""
    recon_images = inference.reconstruct(images)
    
    # 拼接原图和重构图
    num_images = min(8, len(images))
    comparison = torch.cat([
        images[:num_images].cpu(),
        recon_images[:num_images].cpu()
    ])
    
    # 创建网格
    grid = torchvision.utils.make_grid(
        comparison,
        nrow=num_images,
        normalize=True,
        padding=2,
        pad_value=1
    )
    
    # 显示
    plt.figure(figsize=(16, 4.5))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title(f'{title}\nTop: Original | Bottom: Reconstructed', 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Reconstruction saved to {save_path}")
    
    plt.show()


def visualize_generation(
    inference: VAEInference,
    num_samples: int = 16,
    temperature: float = 1.0,
    save_path: Optional[str] = None
):
    """可视化随机生成"""
    samples = inference.generate(num_samples, temperature=temperature)
    
    grid = torchvision.utils.make_grid(
        samples.cpu(),
        nrow=4,
        normalize=True,
        padding=2,
        pad_value=1
    )
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title(f'Random Samples (temperature={temperature:.1f})',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Generated samples saved to {save_path}")
    
    plt.show()


def visualize_interpolation(
    inference: VAEInference,
    image1: torch.Tensor,
    image2: torch.Tensor,
    num_steps: int = 10,
    save_path: Optional[str] = None
):
    """可视化插值 - 同时展示线性和球面插值"""
    # 线性插值
    linear_interp = inference.interpolate(image1, image2, num_steps, mode='linear')
    
    # 球面插值
    spherical_interp = inference.interpolate(image1, image2, num_steps, mode='spherical')
    
    # 拼接
    comparison = torch.cat([linear_interp.cpu(), spherical_interp.cpu()])
    
    grid = torchvision.utils.make_grid(
        comparison,
        nrow=num_steps,
        normalize=True,
        padding=2,
        pad_value=1
    )
    
    plt.figure(figsize=(18, 5))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Latent Space Interpolation\nTop: Linear | Bottom: Spherical (SLERP)',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Interpolation saved to {save_path}")
    
    plt.show()


def visualize_latent_exploration(
    inference: VAEInference,
    image: torch.Tensor,
    num_dims: int = 5,
    save_path: Optional[str] = None
):
    """可视化潜在维度探索"""
    # 选择前num_dims个维度
    dim_indices = list(range(num_dims))
    
    results = inference.explore_latent_dimensions(
        image,
        dim_indices=dim_indices,
        value_range=(-3, 3),
        num_steps=7
    )
    
    grid = torchvision.utils.make_grid(
        results.cpu(),
        nrow=7,
        normalize=True,
        padding=2,
        pad_value=1
    )
    
    plt.figure(figsize=(16, 2.5 * num_dims))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title(f'Latent Dimension Exploration (dims {dim_indices})\n'
              f'Each row shows variation in one dimension from -3 to +3',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Latent exploration saved to {save_path}")
    
    plt.show()


def visualize_temperature_sampling(
    inference: VAEInference,
    temperatures: List[float] = [0.5, 0.8, 1.0, 1.2, 1.5],
    num_samples_per_temp: int = 8,
    save_path: Optional[str] = None
):
    """可视化不同温度下的采样"""
    all_samples = []
    
    for temp in temperatures:
        samples = inference.generate(num_samples_per_temp, temperature=temp)
        all_samples.append(samples)
    
    # 拼接所有样本
    all_samples = torch.cat(all_samples, dim=0)
    
    grid = torchvision.utils.make_grid(
        all_samples.cpu(),
        nrow=num_samples_per_temp,
        normalize=True,
        padding=2,
        pad_value=1
    )
    
    plt.figure(figsize=(18, 2.5 * len(temperatures)))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    
    # 添加温度标签
    title = 'Temperature Effect on Sampling Diversity\n'
    title += ' | '.join([f'T={t:.1f}' for t in temperatures])
    plt.title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Temperature comparison saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='VAE Inference - Optimized',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 运行所有可视化
  python infer_vae_optimized.py --checkpoint ./checkpoints/best.pt
  
  # 只生成样本
  python infer_vae_optimized.py --checkpoint ./checkpoints/best.pt --mode generate
  
  # 探索潜在空间
  python infer_vae_optimized.py --checkpoint ./checkpoints/best.pt --mode latent --vis-method tsne
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='模型检查点路径')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['reconstruct', 'generate', 'interpolate', 
                               'latent', 'explore', 'temperature', 'all'],
                       help='推理模式')
    parser.add_argument('--num-samples', type=int, default=16, 
                       help='生成样本数量')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='生成温度')
    parser.add_argument('--vis-method', type=str, default='pca',
                       choices=['pca', 'tsne', 'umap'],
                       help='潜在空间可视化方法')
    parser.add_argument('--output-dir', type=str, default='./inference_outputs', 
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='设备')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 创建推理器
    print("\n" + "="*60)
    inference = VAEInference(args.checkpoint, args.device)
    print("="*60 + "\n")
    
    # 加载测试数据
    transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )
    
    # 获取测试图像
    test_images, test_labels = next(iter(test_loader))
    
    # 执行推理
    if args.mode in ['reconstruct', 'all']:
        print("\n" + "="*60)
        print("RECONSTRUCTION")
        print("="*60)
        visualize_reconstruction(
            inference,
            test_images[:16],
            save_path=output_dir / 'reconstruction.png'
        )
    
    if args.mode in ['generate', 'all']:
        print("\n" + "="*60)
        print("GENERATION")
        print("="*60)
        visualize_generation(
            inference,
            num_samples=args.num_samples,
            temperature=args.temperature,
            save_path=output_dir / 'generation.png'
        )
    
    if args.mode in ['interpolate', 'all']:
        print("\n" + "="*60)
        print("INTERPOLATION")
        print("="*60)
        visualize_interpolation(
            inference,
            test_images[0],
            test_images[1],
            num_steps=10,
            save_path=output_dir / 'interpolation.png'
        )
    
    if args.mode in ['explore', 'all']:
        print("\n" + "="*60)
        print("LATENT DIMENSION EXPLORATION")
        print("="*60)
        visualize_latent_exploration(
            inference,
            test_images[0],
            num_dims=5,
            save_path=output_dir / 'latent_exploration.png'
        )
    
    if args.mode in ['temperature', 'all']:
        print("\n" + "="*60)
        print("TEMPERATURE SAMPLING")
        print("="*60)
        visualize_temperature_sampling(
            inference,
            temperatures=[0.5, 0.8, 1.0, 1.2, 1.5],
            save_path=output_dir / 'temperature_sampling.png'
        )
    
    if args.mode in ['latent', 'all']:
        print("\n" + "="*60)
        print("LATENT SPACE VISUALIZATION")
        print("="*60)
        inference.visualize_latent_space_2d(
            test_loader,
            num_samples=1000,
            method=args.vis_method,
            save_path=output_dir / f'latent_space_{args.vis_method}.png'
        )
    
    # 计算重构误差
    if args.mode == 'all':
        print("\n" + "="*60)
        print("RECONSTRUCTION ERROR ANALYSIS")
        print("="*60)
        mean_error, errors = inference.get_reconstruction_error(test_loader, num_batches=50)
        print(f"Mean reconstruction error (MSE): {mean_error:.6f}")
        print(f"Std: {errors.std():.6f}")
        print(f"Min: {errors.min():.6f}")
        print(f"Max: {errors.max():.6f}")
        
        # 可视化误差分布
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Reconstruction Error (MSE)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Distribution of Reconstruction Errors', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Error distribution saved to {output_dir / 'error_distribution.png'}")
        plt.show()
    
    print("\n" + "="*60)
    print(f"✓ All outputs saved to {output_dir}")
    print("="*60)

# python inference.py --checkpoint ./checkpoints/best.pt
if __name__ == '__main__':
    main()