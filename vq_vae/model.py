import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """残差块 - 保持通道数不变"""

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


class VectorQuantizer(nn.Module):
    """
    VQ layer with proper straight-through estimator
    
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 commitment_cost: float = 0.25,
                 use_ema: bool = True,
                 ema_decay: float = 0.99):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # 初始化码本
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        # EMA参数
        if use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: (B, C, H, W) 编码器输出的连续向量

        Returns:
            z_q: (B, C, H, W) 量化后的向量
            loss: vq损失
            indices: (B, H, W) 码本索引
        """
        B, C, H, W = z_e.shape

        # 展平以计算距离
        z = z_e.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        z_flattened = z.view(-1, C)  # (BHW, C)

        # ========== 距离计算 ==========
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
                z_flattened.pow(2).sum(dim=1, keepdim=True)  # (BHW, 1)
                - 2 * z_flattened @ self.embedding.weight.t()  # (BHW, K)
                + self.embedding.weight.pow(2).sum(dim=1)  # (K,)
        )

        # 量化（找最近邻）
        encoding_indices = torch.argmin(distances, dim=1)  # (BHW,)
        z_q_flat = self.embedding(encoding_indices)  # (BHW, C)
        z_q = z_q_flat.view(z.shape)  # (B, H, W, C)
        # ========== 计算损失 ==========        
        if self.use_ema:
            # EMA模式：只计算commitment loss，码本通过EMA更新
            # commitment loss: 让编码器输出靠近码本
            commitment_loss = F.mse_loss(z, z_q.detach())
            loss = self.commitment_cost * commitment_loss
            
            # 用于监控的vq_loss（不参与梯度）
            with torch.no_grad():
                vq_loss_monitor = F.mse_loss(z_q, z)
        else:
            # 梯度模式：两个loss都计算梯度
            # vq_loss: 让码本向编码器输出靠近（对码本有梯度）
            vq_loss = F.mse_loss(z_q, z.detach())
            
            # commitment_loss: 让编码器输出靠近码本（对编码器有梯度）
            commitment_loss = F.mse_loss(z, z_q.detach())
            
            loss = vq_loss + self.commitment_cost * commitment_loss

        # ========== 码本更新 ==========
        if self.training and self.use_ema:
            self._update_ema(z_flattened, encoding_indices)

        # ========== Straight-Through Estimator ==========
        # 前向：z_q（量化后的离散值）
        # 反向：梯度从z_q直接流到z（绕过量化）
        z_q_ste = z + (z_q - z).detach()

        # 恢复形状
        z_q_ste = z_q_ste.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return z_q_ste, loss, encoding_indices.view(B, H, W)

    def _update_ema(self, z_flattened: torch.Tensor, encoding_indices: torch.Tensor):
        """
        使用指数移动平均(EMA)更新码本向量
        这比梯度更新更稳定
        """
        with torch.no_grad():
            # 计算每个码本向量被使用的次数
            encodings_onehot = F.one_hot(encoding_indices, self.num_embeddings).float()  # (BHW, K)
            
            # 更新聚类大小
            cluster_size = encodings_onehot.sum(0)  # (K,)
            updated_cluster_size = (
                self.ema_decay * self.ema_cluster_size +
                (1 - self.ema_decay) * cluster_size
            )
            
            # Laplace smoothing
            n = updated_cluster_size.sum()
            updated_cluster_size = (
                (updated_cluster_size + 1e-5) /
                (n + self.num_embeddings * 1e-5) * n
            )
            
            # 更新嵌入向量的累积和
            dw = encodings_onehot.t() @ z_flattened  # (K, C)
            updated_w = (
                self.ema_decay * self.ema_w +
                (1 - self.ema_decay) * dw
            )
            
            # 归一化得到新的码本向量
            self.embedding.weight.data = updated_w / updated_cluster_size.unsqueeze(1)
            
            # 更新buffer
            self.ema_w.copy_(updated_w)
            self.ema_cluster_size.copy_(updated_cluster_size)


class DeepConvVQVAE(nn.Module):
    """
    VQ-VAE with proper architecture
    """

    def __init__(
            self,
            in_channels: int = 1,
            image_size: int = 28,
            hidden_channels: list = [32, 64],
            embedding_dim: int = 64,  # 每个码本向量的维度，类比：每个"单词"用64维向量表示
            num_embeddings: int = 512, # 码本中有多少个向量 类比：词典里有512个"单词" 图像会被编码成这512种"单词"的组合
            num_res_blocks: int = 2,
            commitment_cost: float = 0.25,
            use_ema: bool = True,
            ema_decay: float = 0.99
    ):
        super().__init__()

        self.in_channels = in_channels
        self.image_size = image_size
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # ================= Encoder =================
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

        # 编码器后的残差块
        self.encoder_res = nn.Sequential(
            *[ResidualBlock(hidden_channels[-1]) for _ in range(num_res_blocks)]
        )

        # 投影到embedding维度
        self.conv_to_embedding = nn.Conv2d(
            hidden_channels[-1],
            embedding_dim,
            kernel_size=1
        )

        # ================= VQ Layer =================
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            use_ema=use_ema,
            ema_decay=ema_decay
        )

        # ================= Decoder =================
        # 解码器前的残差块
        self.decoder_res = nn.Sequential(
            *[ResidualBlock(embedding_dim) for _ in range(num_res_blocks)]
        )

        # 逐层上采样
        decoder_layers = []
        reversed_channels = list(reversed(hidden_channels))

        # 第一层：embedding_dim -> reversed_channels[0]
        decoder_layers.append(
            nn.Conv2d(embedding_dim, reversed_channels[0], kernel_size=3, padding=1)
        )

        # 中间层：上采样
        for i in range(len(reversed_channels) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(
                    reversed_channels[i],
                    reversed_channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(reversed_channels[i + 1]),
                nn.ReLU()
            ])

        # 最后一层
        decoder_layers.extend([
            nn.ConvTranspose2d(
                reversed_channels[-1],
                in_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        ])

        self.decoder = nn.Sequential(*decoder_layers)

    # ================= Forward Methods =================
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码到连续空间"""
        h = self.encoder(x)
        h = self.encoder_res(h)
        z_e = self.conv_to_embedding(h)
        return z_e

    def quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """量化到离散码本"""
        z_q, vq_loss, indices = self.vq(z_e)
        return z_q, vq_loss, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """从量化表示解码"""
        h = self.decoder_res(z_q)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """完整前向传播"""
        z_e = self.encode(x)
        z_q, vq_loss, indices = self.quantize(z_e)
        x_recon = self.decode(z_q)
        return x_recon, vq_loss, indices

    # ================= Utilities =================
    @torch.no_grad()
    def sample(self, indices: torch.Tensor) -> torch.Tensor:
        """
        从码本索引生成图像
        
        Args:
            indices: (B, H, W) 码本索引
            
        Returns:
            图像: (B, C, H, W)
        """
        z_q = self.vq.embedding(indices).permute(0, 3, 1, 2)  # (B, C, H, W)
        return self.decode(z_q)
    
    @torch.no_grad()
    def get_code_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取输入图像的码本索引（用于分类）
        
        Args:
            x: (B, C, H, W) 输入图像
            
        Returns:
            indices: (B, H, W) 码本索引
        """
        z_e = self.encode(x)
        _, _, indices = self.quantize(z_e)
        return indices

    def get_codebook_usage(self) -> float:
        """获取码本使用率"""
        if hasattr(self.vq, 'ema_cluster_size'):
            return (self.vq.ema_cluster_size > 0).float().mean().item()
        return None

    def get_perplexity(self) -> Optional[float]:
        """计算码本困惑度"""
        if hasattr(self.vq, 'ema_cluster_size'):
            ps = self.vq.ema_cluster_size / self.vq.ema_cluster_size.sum()
            ps = ps.clamp(min=1e-10)
            entropy = -(ps * torch.log(ps)).sum()
            perplexity = torch.exp(entropy).item()
            return perplexity
        return None


# ==================== 测试代码 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 创建模型
    model = DeepConvVQVAE(
        in_channels=1,
        image_size=28,
        hidden_channels=[32, 64],
        embedding_dim=64,
        num_embeddings=512,
        num_res_blocks=2,
        commitment_cost=0.25,
        use_ema=True,
        ema_decay=0.99
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28).to(device)
    
    model.train()
    x_recon, vq_loss, indices = model(x)

    print(f"\n=== Forward Pass ===")
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Indices shape: {indices.shape}")
    
    if model.get_codebook_usage():
        print(f"Codebook usage: {model.get_codebook_usage():.2%}")
        print(f"Perplexity: {model.get_perplexity():.2f}")

    # 测试重构损失
    recon_loss = F.mse_loss(x_recon, x)
    total_loss = recon_loss + vq_loss

    print(f"\n=== Losses ===")
    print(f"Reconstruction Loss: {recon_loss.item():.4f}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")

    # 测试采样
    print(f"\n=== Sampling ===")
    sample_indices = torch.randint(0, 512, (batch_size, 7, 7)).to(device)
    samples = model.sample(sample_indices)
    print(f"Sampled image shape: {samples.shape}")
    
    # 测试获取索引
    print(f"\n=== Get Code Indices ===")
    code_indices = model.get_code_indices(x)
    print(f"Code indices shape: {code_indices.shape}")