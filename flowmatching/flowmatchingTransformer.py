import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimestepEmbedding(nn.Module):
    """时间步嵌入 - 使用正弦位置编码"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2

        # 预计算频率
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        """
        t: [B] 时间步 (0到1之间)
        return: [B, dim]
        """
        args = t[:, None].float() * self.freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class LabelEmbedding(nn.Module):
    """类别嵌入"""

    def __init__(self, num_classes, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, dim)

    def forward(self, y):
        return self.embedding(y)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization - DiT 的核心创新"""

    def __init__(self, normalized_shape, cond_dim):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, elementwise_affine=False)

        # 从条件生成 scale 和 shift
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * normalized_shape)
        )

        # 零初始化，使初始时 AdaLN ≈ LN
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, cond):
        """
        x: [B, N, D] 输入特征
        cond: [B, D] 条件向量
        """
        # LayerNorm
        x = self.ln(x)

        # 生成 scale 和 shift
        modulation = self.adaLN_modulation(cond)  # [B, 2*D]
        scale, shift = modulation.chunk(2, dim=-1)  # [B, D], [B, D]

        # 应用仿射变换
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return x


class Attention(nn.Module):
    """多头自注意力"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV 投影
        self.qkv = nn.Linear(dim, dim * 3)

        # 输出投影
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: [B, N, D]
        return: [B, N, D]
        """
        B, N, D = x.shape

        # 计算 QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = F.softmax(attn, dim=-1)

        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]

        # 输出投影
        x = self.proj(x)

        return x


class MLP(nn.Module):
    """前馈网络"""

    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate='tanh')  # 使用近似 GELU
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DiTBlock(nn.Module):
    """DiT Block - 使用 AdaLN 的 Transformer Block"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()

        # AdaLN for attention
        self.adaLN_attn = AdaLN(dim, dim)
        self.attn = Attention(dim, num_heads)

        # AdaLN for MLP
        self.adaLN_mlp = AdaLN(dim, dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x, cond):
        """
        x: [B, N, D] patch tokens
        cond: [B, D] 条件向量 (融合了时间和类别)
        """
        # 注意力分支
        x = x + self.attn(self.adaLN_attn(x, cond))

        # MLP 分支
        x = x + self.mlp(self.adaLN_mlp(x, cond))

        return x


class FlowMatchingTransformer(nn.Module):
    """基于 Transformer 的 Flow Matching 模型 (DiT 风格)"""

    def __init__(
            self,
            img_size=28,  # 输入图像尺寸
            patch_size=4,  # patch 大小
            in_channels=1,  # 输入通道数
            hidden_dim=384,  # Transformer 维度
            depth=12,  # Transformer 层数
            num_heads=6,  # 注意力头数
            mlp_ratio=4.0,  # MLP 扩展比例
            num_classes=10,  # 类别数
            class_dropout_prob=0.1  # CFG dropout 概率
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.num_patches = (img_size // patch_size) ** 2

        # ============ 输入层 ============

        # Patchify: 将图像切分成 patches
        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 位置编码 (可学习)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_dim)
        )

        # ============ 条件编码 ============

        # 时间步编码
        self.time_embed = nn.Sequential(
            TimestepEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 类别编码 (num_classes + 1 for unconditional)
        self.label_embed = LabelEmbedding(num_classes + 1, hidden_dim)

        # 融合时间和类别
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ============ Transformer Blocks ============

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # ============ 输出层 ============

        # 最终的 AdaLN
        self.final_adaLN = AdaLN(hidden_dim, hidden_dim)

        # 投影回 patch 空间
        self.final_layer = nn.Linear(
            hidden_dim,
            patch_size * patch_size * in_channels
        )

        # 小方差初始化输出层
        nn.init.normal_(self.final_layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.final_layer.bias)

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        """初始化权重"""
        # 位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 其他线性层
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def patchify(self, x):
        """
        将图像切分成 patches
        x: [B, C, H, W]
        return: [B, N, patch_size^2 * C]
        """
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x

    def unpatchify(self, x):
        """
        将 patches 还原成图像
        x: [B, N, patch_size^2 * C]
        return: [B, C, H, W]
        """
        B = x.shape[0]
        h = w = self.img_size // self.patch_size
        p = self.patch_size
        c = self.in_channels

        x = x.reshape(B, h, w, p, p, c)
        x = torch.einsum('bhwpqc->bchpwq', x)
        x = x.reshape(B, c, h * p, w * p)

        return x

    def forward(self, x, t, y=None):
        """
        x: [B, C, H, W] 当前状态
        t: [B] 时间步 (0到1)
        y: [B] 类别标签 (可选)
        return: [B, C, H, W] 预测的速度场
        """
        B = x.shape[0]

        # ============ 1. Patch Embedding ============
        x = self.patchify(x)  # [B, N, D]

        # ============ 2. 位置编码 ============
        x = x + self.pos_embed

        # ============ 3. 条件编码 ============

        # 时间步编码
        t_emb = self.time_embed(t)  # [B, D]

        # 类别编码 (Classifier-Free Guidance)
        if y is None:
            # 无条件生成
            y = torch.full(
                (B,), self.num_classes,
                device=x.device, dtype=torch.long
            )

        # 训练时随机 dropout 类别 (CFG)
        if self.training:
            dropout_mask = torch.rand(B, device=x.device) < self.class_dropout_prob
            y = torch.where(dropout_mask, self.num_classes, y)

        y_emb = self.label_embed(y)  # [B, D]

        # 融合条件
        cond = self.cond_mlp(torch.cat([t_emb, y_emb], dim=1))  # [B, D]

        # ============ 4. Transformer Blocks ============
        for block in self.blocks:
            x = block(x, cond)

        # ============ 5. 输出层 ============
        x = self.final_adaLN(x, cond)  # 最终归一化
        x = self.final_layer(x)  # [B, N, P*P*C]

        # ============ 6. Unpatchify ============
        v = self.unpatchify(x)  # [B, C, H, W]

        return v


def check_model():
    """测试模型"""
    print("=" * 80)
    print("Flow Matching Transformer 模型测试")
    print("=" * 80)

    # 创建模型
    model = FlowMatchingTransformer(
        img_size=28,
        patch_size=4,
        in_channels=1,
        hidden_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=10
    ).cuda()

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"  总参数: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")

    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28).cuda()
    t = torch.rand(batch_size).cuda()
    y = torch.randint(0, 10, (batch_size,)).cuda()

    print(f"\n前向传播测试:")
    with torch.no_grad():
        v = model(x, t, y)

    print(f"\n输入统计:")
    print(f"  x: shape={x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"  t: shape={t.shape}, range=[{t.min():.4f}, {t.max():.4f}]")
    print(f"  y: shape={y.shape}")

    print(f"\n输出统计:")
    print(f"  v: shape={v.shape}, mean={v.mean():.4f}, std={v.std():.4f}")
    print(f"  v: min={v.min():.4f}, max={v.max():.4f}")
    print(f"  v: norm={v.norm():.4f}")

    # 检查输出是否正常
    if v.abs().max() < 1e-6:
        print(f"\n⚠️  警告: 输出接近全零!")
        return False
    elif v.abs().mean() < 0.001:
        print(f"\n⚠️  输出幅度较小，但可接受")
        return True
    else:
        print(f"\n✓ 模型输出正常")
        return True


if __name__ == '__main__':
    check_model()