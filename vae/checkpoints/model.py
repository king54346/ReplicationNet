"""
VAE (Variational Autoencoder) Model Implementation
支持灵活的网络架构配置
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class VAE(nn.Module):
    """变分自编码器（全连接版本）"""

    def __init__(
            self,
            input_dim: int = 784,
            hidden_dims: list = [512, 256],
            latent_dim: int = 20,
            activation: str = 'relu'
    ):
        """
        Args:
            input_dim: 输入维度 (例如28*28=784)
            hidden_dims: 隐藏层维度列表
            latent_dim: 潜在空间维度
            activation: 激活函数类型
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 激活函数选择
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # ==================== Encoder ====================
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # 潜在空间参数
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # ==================== Decoder ====================
        decoder_layers = []

        decoder_layers.extend([
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            self.activation
        ])

        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i - 1]),
                nn.BatchNorm1d(hidden_dims[i - 1]),
                self.activation,
                nn.Dropout(0.2)
            ])

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        编码器：将输入映射到潜在空间的分布参数

        Args:
            x: 输入张量 [batch_size, input_dim]

        Returns:
            mu: 均值 [batch_size, latent_dim]
            logvar: 对数方差 [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧：z = μ + σ * ε, 其中 ε ~ N(0,1)

        为什么这样做？
        1. 直接从 N(μ,σ²) 采样不可微，无法反向传播
        2. 将随机性分离到 ε，使梯度可以通过 μ 和 σ 传播

        数学推导：
        - σ = exp(log_var / 2) = exp(0.5 * log(σ²)) = sqrt(σ²)
        - z = μ + ε·σ 保持分布不变，但可微

        Args:
            mu: 均值张量
            logvar: 对数方差张量

        Returns:
            z: 重参数化后的潜在向量
        """
        if self.training:
            # 训练模式：添加随机性
            std = torch.exp(0.5 * logvar)  # 标准差 = exp(log_var/2)
            eps = torch.randn_like(std)  # 从标准正态分布采样
            return mu + eps * std
        else:
            # 推理模式：返回均值（确定性）
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码器：从潜在空间重构输入

        Args:
            z: 潜在向量 [batch_size, latent_dim]

        Returns:
            recon: 重构输出 [batch_size, input_dim]
        """
        return torch.sigmoid(self.decoder(z))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, input_dim]

        Returns:
            recon_x: 重构输出
            mu: 潜在分布均值
            logvar: 潜在分布对数方差
        """
        # Flatten输入
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # 编码
        mu, logvar = self.encode(x_flat)

        # 重参数化采样
        z = self.reparameterize(mu, logvar)

        # 解码
        recon_x = self.decode(z)

        return recon_x, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        从先验分布 N(0,1) 采样生成新样本

        Args:
            num_samples: 生成样本数量
            device: 设备

        Returns:
            samples: 生成的样本 [num_samples, input_dim]
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples

    def interpolate(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            num_steps: int = 10
    ) -> torch.Tensor:
        """
        在两个样本之间进行潜在空间插值

        Args:
            x1: 起始样本
            x2: 终止样本
            num_steps: 插值步数

        Returns:
            interpolations: 插值结果 [num_steps, input_dim]
        """
        self.eval()
        with torch.no_grad():
            # 编码到潜在空间
            mu1, _ = self.encode(x1.view(1, -1))
            mu2, _ = self.encode(x2.view(1, -1))

            # 线性插值
            alphas = torch.linspace(0, 1, num_steps).to(x1.device)
            interpolations = []

            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decode(z_interp)
                interpolations.append(x_interp)

            return torch.cat(interpolations, dim=0)


def vae_loss(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0
) -> tuple[torch.Tensor, dict]:
    """
    VAE损失函数 = 重构损失 + KL散度损失

    数学原理：
    1. 重构损失：衡量重建质量
       BCE = -Σ[x*log(x̂) + (1-x)*log(1-x̂)]

    2. KL散度：约束后验分布 q(z|x) 接近先验分布 p(z)=N(0,1)
       对于正态分布，KL散度有闭式解：
       KL[N(μ,σ²)||N(0,1)] = -0.5 * Σ[1 + log(σ²) - μ² - σ²]

    Args:
        recon_x: 重构输出 [batch_size, ...] (值域应在 [0, 1])
        x: 原始输入 [batch_size, ...] (值域应在 [0, 1])
        mu: 潜在分布均值
        logvar: 潜在分布对数方差
        kl_weight: KL散度权重（用于KL退火）

    Returns:
        loss: 总损失
        loss_dict: 损失组件字典
    """
    batch_size = x.size(0)

    # 数据范围检查（添加警告而不是抛出异常）
    x_flat = x.view(batch_size, -1)
    recon_flat = recon_x.view(batch_size, -1)
    
    if (x_flat.min() < 0 or x_flat.max() > 1):
        warnings.warn(
            f"Input x is outside [0, 1]: min={x_flat.min():.4f}, max={x_flat.max():.4f}. "
            f"Clamping to [0, 1] for BCE calculation.",
            RuntimeWarning
        )
        x_flat = torch.clamp(x_flat, 0, 1)
    
    if (recon_flat.min() < 0 or recon_flat.max() > 1):
        warnings.warn(
            f"Reconstruction is outside [0, 1]: min={recon_flat.min():.4f}, max={recon_flat.max():.4f}. "
            f"Clamping to [0, 1] for BCE calculation.",
            RuntimeWarning
        )
        recon_flat = torch.clamp(recon_flat, 0, 1)

    # 重构损失 (Binary Cross Entropy)
    # 添加数值稳定性: clamp 避免 log(0)
    recon_flat = torch.clamp(recon_flat, 1e-7, 1 - 1e-7)
    recon_loss = F.binary_cross_entropy(
        recon_flat,
        x_flat,
        reduction='sum'
    ) / batch_size

    # KL散度损失
    # KL[N(μ,σ²)||N(0,1)] = -0.5 * Σ[1 + log(σ²) - μ² - σ²]
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    # 总损失
    total_loss = recon_loss + kl_weight * kl_loss

    loss_dict = {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'kl_weight': kl_weight
    }

    return total_loss, loss_dict


class ConvVAE(nn.Module):
    """卷积变分自编码器

    关键修复：正确处理任意尺寸图像的下采样和上采样
    """

    def __init__(
            self,
            in_channels: int = 1,
            image_size: int = 28,
            hidden_channels: list = [32, 64],
            latent_dim: int = 20,
            activation: str = 'relu'
    ):
        super().__init__()

        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # ==================== Encoder ====================
        # 使用 stride=2 的卷积进行下采样
        encoder_layers = []
        prev_channels = in_channels

        # 追踪每层的尺寸，用于解码器的精确上采样
        self.encoder_sizes = [image_size]
        current_size = image_size

        for h_ch in hidden_channels:
            encoder_layers.extend([
                nn.Conv2d(prev_channels, h_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_ch),
                self.activation
            ])
            # 计算输出尺寸: floor((size + 2*padding - kernel) / stride + 1)
            current_size = (current_size + 2 * 1 - 3) // 2 + 1
            self.encoder_sizes.append(current_size)
            prev_channels = h_ch

        self.encoder = nn.Sequential(*encoder_layers)

        # 计算展平后的维度
        self.feature_size = current_size
        self.flatten_dim = hidden_channels[-1] * self.feature_size * self.feature_size

        # 潜在空间参数
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # ==================== Decoder ====================
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        # 构建解码器层（需要精确恢复尺寸）
        decoder_layers = []
        reversed_channels = list(reversed(hidden_channels))
        reversed_sizes = list(reversed(self.encoder_sizes))

        for i in range(len(reversed_channels) - 1):
            # 目标尺寸是 reversed_sizes[i+1]（编码器对应层的输入尺寸）
            target_size = reversed_sizes[i + 1]
            decoder_layers.extend([
                nn.ConvTranspose2d(
                    reversed_channels[i],
                    reversed_channels[i + 1],
                    kernel_size=3, stride=2, padding=1,
                    output_padding=self._compute_output_padding(reversed_sizes[i], target_size)
                ),
                nn.BatchNorm2d(reversed_channels[i + 1]),
                self.activation
            ])

        # 输出层
        target_size = reversed_sizes[-1]  # 原始图像尺寸
        current_decoder_size = reversed_sizes[-2] if len(reversed_sizes) > 1 else self.feature_size
        decoder_layers.append(
            nn.ConvTranspose2d(
                reversed_channels[-1],
                in_channels,
                kernel_size=3, stride=2, padding=1,
                output_padding=self._compute_output_padding(current_decoder_size, target_size)
            )
        )

        self.decoder = nn.Sequential(*decoder_layers)

    def _compute_output_padding(self, input_size: int, target_size: int) -> int:
        """计算转置卷积所需的 output_padding

        转置卷积输出尺寸公式:
        out = (in - 1) * stride - 2 * padding + kernel + output_padding

        对于 stride=2, padding=1, kernel=3:
        out = (in - 1) * 2 - 2 + 3 + output_padding = 2*in - 1 + output_padding

        所以: output_padding = target - (2*input - 1)
        """
        base_output = 2 * input_size - 1
        output_padding = target_size - base_output
        return max(0, min(output_padding, 1))  # 限制在 [0, 1] 范围内

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧：z = μ + σ * ε, 其中 ε ~ N(0,1)

        为什么这样做？
        1. 直接从 N(μ,σ²) 采样不可微，无法反向传播
        2. 将随机性分离到 ε，使梯度可以通过 μ 和 σ 传播

        数学推导：
        - σ = exp(log_var / 2) = exp(0.5 * log(σ²)) = sqrt(σ²)
        - z = μ + ε·σ 保持分布不变，但可微

        Args:
            mu: 均值张量
            logvar: 对数方差张量

        Returns:
            z: 重参数化后的潜在向量
        """
        if self.training:
            # 训练模式：添加随机性
            std = torch.exp(0.5 * logvar)  # 标准差 = exp(log_var/2)
            eps = torch.randn_like(std)  # 从标准正态分布采样
            return mu + eps * std
        else:
            # 推理模式：返回均值（确定性）
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self.hidden_channels[-1], self.feature_size, self.feature_size)
        recon = self.decoder(h)
        return torch.sigmoid(recon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            if x1.dim() == 3:
                x1 = x1.unsqueeze(0)
            if x2.dim() == 3:
                x2 = x2.unsqueeze(0)

            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)

            alphas = torch.linspace(0, 1, num_steps).to(x1.device)
            interpolations = []

            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decode(z_interp)
                interpolations.append(x_interp)

            return torch.cat(interpolations, dim=0)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            return recon


def conv_vae_loss(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0
) -> tuple[torch.Tensor, dict]:
    """ConvVAE 损失函数
    
    Args:
        recon_x: 重构输出
        x: 原始输入
        mu: 潜在分布均值
        logvar: 潜在分布对数方差
        kl_weight: KL散度权重
    """
    batch_size = x.size(0)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    total_loss = recon_loss + kl_weight * kl_loss

    loss_dict = {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'kl_weight': kl_weight
    }

    return total_loss, loss_dict


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class DeepConvVAE(nn.Module):
    """带残差连接的深层卷积VAE"""

    def __init__(
            self,
            in_channels: int = 1,
            image_size: int = 28,
            hidden_channels: list = [32, 64],
            latent_dim: int = 32,
            num_res_blocks: int = 2
    ):
        super().__init__()

        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        # ==================== Encoder ====================
        encoder_layers = []
        prev_channels = in_channels

        for h_ch in hidden_channels:
            encoder_layers.extend([
                nn.Conv2d(prev_channels, h_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_ch),
                nn.ReLU()
            ])
            prev_channels = h_ch

        self.encoder = nn.Sequential(*encoder_layers)

        # 残差块
        self.encoder_res = nn.Sequential(
            *[ResidualBlock(hidden_channels[-1]) for _ in range(num_res_blocks)]
        )

        # 动态计算特征图尺寸
        self.feature_size = self._get_conv_output_size(image_size, len(hidden_channels))
        self.flatten_dim = hidden_channels[-1] * self.feature_size * self.feature_size

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # ==================== Decoder ====================
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder_res = nn.Sequential(
            *[ResidualBlock(hidden_channels[-1]) for _ in range(num_res_blocks)]
        )

        decoder_layers = []
        reversed_channels = list(reversed(hidden_channels))
        sizes = self._get_decoder_sizes(image_size, len(hidden_channels))

        for i in range(len(reversed_channels) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(
                    reversed_channels[i],
                    reversed_channels[i + 1],
                    kernel_size=4, stride=2, padding=1,
                    output_padding=self._get_output_padding(sizes[i], sizes[i + 1])
                ),
                nn.BatchNorm2d(reversed_channels[i + 1]),
                nn.ReLU()
            ])

        decoder_layers.append(
            nn.ConvTranspose2d(
                reversed_channels[-1],
                in_channels,
                kernel_size=4, stride=2, padding=1,
                output_padding=self._get_output_padding(sizes[-2], sizes[-1])
            )
        )

        self.decoder = nn.Sequential(*decoder_layers)

    def _get_conv_output_size(self, size: int, num_layers: int) -> int:
        """计算卷积层输出尺寸"""
        for _ in range(num_layers):
            size = (size + 2 * 1 - 4) // 2 + 1
        return size

    def _get_decoder_sizes(self, target_size: int, num_layers: int) -> list:
        """计算解码器各层的目标尺寸"""
        sizes = [target_size]
        size = target_size
        for _ in range(num_layers):
            size = (size + 2 * 1 - 4) // 2 + 1
            sizes.insert(0, size)
        return sizes

    def _get_output_padding(self, input_size: int, target_size: int) -> int:
        """计算转置卷积的output_padding"""
        expected = (input_size - 1) * 2 - 2 * 1 + 4
        return max(0, min(target_size - expected, 1))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = self.encoder_res(h)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self.hidden_channels[-1], self.feature_size, self.feature_size)
        h = self.decoder_res(h)
        recon = self.decoder(h)
        return torch.sigmoid(recon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples

    def interpolate(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            num_steps: int = 10
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            if x1.dim() == 3:
                x1 = x1.unsqueeze(0)
            if x2.dim() == 3:
                x2 = x2.unsqueeze(0)

            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)

            alphas = torch.linspace(0, 1, num_steps).to(x1.device)
            interpolations = []

            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decode(z_interp)
                interpolations.append(x_interp)

            return torch.cat(interpolations, dim=0)


if __name__ == "__main__":
    print("=" * 50)
    print("Testing VAE")
    print("=" * 50)
    model = VAE(input_dim=784, hidden_dims=[512, 256], latent_dim=20)
    # 修复: 使用 [0, 1] 范围的随机数据
    x = torch.rand(32, 1, 28, 28)  # 使用 rand 而不是 randn
    recon_x, mu, logvar = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Recon shape: {recon_x.shape}")
    print(f"Recon range: [{recon_x.min():.4f}, {recon_x.max():.4f}]")
    loss, loss_dict = vae_loss(recon_x, x, mu, logvar)
    print(f"Loss: {loss_dict}")

    print("\n" + "=" * 50)
    print("Testing ConvVAE")
    print("=" * 50)
    conv_model = ConvVAE(in_channels=1, image_size=28, hidden_channels=[32, 64], latent_dim=20)
    x = torch.rand(32, 1, 28, 28)  # 修复: 使用 rand
    recon_x, mu, logvar = conv_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Recon shape: {recon_x.shape}")
    print(f"Recon range: [{recon_x.min():.4f}, {recon_x.max():.4f}]")
    loss, loss_dict = conv_vae_loss(recon_x, x, mu, logvar)
    print(f"Loss: {loss_dict}")

    print("\n" + "=" * 50)
    print("Testing DeepConvVAE")
    print("=" * 50)
    deep_model = DeepConvVAE(in_channels=1, image_size=28, hidden_channels=[32, 64], latent_dim=32)
    x = torch.rand(32, 1, 28, 28)  # 修复: 使用 rand
    recon_x, mu, logvar = deep_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Recon shape: {recon_x.shape}")
    print(f"Recon range: [{recon_x.min():.4f}, {recon_x.max():.4f}]")
    loss, loss_dict = conv_vae_loss(recon_x, x, mu, logvar)
    print(f"Loss: {loss_dict}")

    # 验证尺寸匹配
    assert x.shape == recon_x.shape, f"Shape mismatch: {x.shape} vs {recon_x.shape}"
    print("\n✓ All tests passed!")