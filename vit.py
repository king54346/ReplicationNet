import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    将图像切分成patches并进行embedding

    例如: 224×224×3 的图像, patch_size=16
         → 切成 14×14=196 个 16×16×3 的patch
         → 每个patch flatten成 768 维向量
         → 输出: [B, 196, 768]
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 方法1: 使用卷积实现 (推荐)
        # 等价于: 切patch → flatten → linear projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x: [B, C, H, W] 例如 [B, 3, 224, 224]
        return: [B, num_patches, embed_dim] 例如 [B, 196, 768]
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input size ({H}×{W}) doesn't match model ({self.img_size}×{self.img_size})"

        # 卷积投影: [B, 3, 224, 224] → [B, 768, 14, 14]
        x = self.proj(x)

        # Flatten并转置: [B, 768, 14, 14] → [B, 768, 196] → [B, 196, 768]
        x = x.flatten(2).transpose(1, 2)

        return x


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    """

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV 投影 (一次性计算)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)

        # 输出投影
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, N, D] 其中 N = num_patches + 1 (包含CLS token)
        """
        B, N, D = x.shape

        # === 1. 计算 Q, K, V ===
        # [B, N, D] → [B, N, 3D] → [B, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        # [B, N, 3, H, d] → [3, B, H, N, d]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 分离 Q, K, V: 每个是 [B, H, N, d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # === 2. 计算注意力分数 ===
        # [B, H, N, d] @ [B, H, d, N] → [B, H, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Softmax
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # === 3. 加权求和 ===
        # [B, H, N, N] @ [B, H, N, d] → [B, H, N, d]
        out = attn @ v

        # === 4. 合并多头 ===
        # [B, H, N, d] → [B, N, H, d] → [B, N, D]
        out = out.transpose(1, 2).reshape(B, N, D)

        # === 5. 输出投影 ===
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out


# 输入 [B, N, 768]
#     ↓
# Linear (768 → 3072)  ← "扩展" 到更高维度
#     ↓
# GELU 激活             ← 非线性变换
#     ↓
# Dropout (p=0.1)      ← 随机失活 (训练时)
#     ↓
# Linear (3072 → 768)  ← "压缩" 回原始维度
#     ↓
# Dropout (p=0.1)
#     ↓
# 输出 [B, N, 768]
class MLP(nn.Module):
    """
    前馈神经网络 (Feed-Forward Network)
    结构: Linear → GELU → Dropout → Linear → Dropout
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x



# 信息流:
# x → Norm → Attn → + → Norm → MLP → +
# ↓                  ↑  ↓              ↑
# └──────────────────┘  └──────────────┘
# nn.TransformerEncoder: Post-Norm
# 信息流:
# x → Attn → + → Norm → MLP → + → Norm
# ↓          ↑          ↓         ↑
# └──────────┘          └─────────┘
class TransformerBlock(nn.Module):
    """
    Transformer编码器块
    结构:
      x → LayerNorm → MHSA → + → LayerNorm → MLP → +
      ↓                        ↑  ↓                  ↑
      └────────────────────────┘  └──────────────────┘
    """

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        # Layer Normalization (Pre-Norm)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Multi-Head Self-Attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        # Layer Normalization
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        # Attention block (残差连接)
        x = x + self.attn(self.norm1(x))

        # MLP block (残差连接)
        x = x + self.mlp(self.norm2(x))

        return x




# 输入图像 [B, 3, 224, 224]
#     ↓
# Patch Embedding [B, 196, 768]
#     ↓
# 添加 CLS Token [B, 197, 768]
#     ↓
# 添加 Position Embedding
#     ↓
# Transformer Block 1
#     ↓
# Transformer Block 2
#     ↓
#     ...
#     ↓
# Transformer Block 12
#     ↓
# Layer Norm
#     ↓
# 提取 CLS Token [B, 768]
#     ↓
# Classification Head [B, 1000]
class ViT(nn.Module):
    """
    Vision Transformer

    论文: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    https://arxiv.org/abs/2010.11929

    Args:
        img_size: 输入图像大小 (默认224)
        patch_size: patch大小 (默认16)
        in_channels: 输入通道数 (默认3, RGB)
        num_classes: 分类类别数 (默认1000, ImageNet)
        embed_dim: embedding维度 (默认768)
        depth: Transformer层数 (默认12)
        num_heads: 注意力头数 (默认12)
        mlp_ratio: MLP隐层维度倍数 (默认4.0)
        dropout: Dropout概率 (默认0.0)
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=0.0,
            emb_dropout=0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # === 1. Patch Embedding ===
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # === 2. CLS Token (可学习的分类token) ===
        # 在序列最前面添加一个特殊token,用于分类
        # 添加 CLS Token 后，借鉴自 BERT 的 [CLS] token，包含了整个图像的语义信息，通过线性分类头进行分类
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # === 3. Position Embedding (可学习的位置编码) ===
        # num_patches + 1 是因为要包含 cls_token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_dropout = nn.Dropout(emb_dropout)

        # === 4. Transformer Blocks ===
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        # === 5. Layer Norm ===
        self.norm = nn.LayerNorm(embed_dim)

        # === 6. Classification Head ===
        # 手写数字10分类
        self.head = nn.Linear(embed_dim, num_classes)

        # === 7. 初始化权重 ===
        self.initialize_weights()

    def initialize_weights(self):
        """初始化权重"""
        # Position embedding 用截断正态分布
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 其他层用标准初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化函数"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """
        提取特征 (不包括分类头)
        x: [B, 3, 224, 224]
        return: [B, embed_dim]
        """
        B = x.shape[0]

        # === 1. Patch Embedding ===
        # [B, 3, 224, 224] → [B, 196, 768]
        x = self.patch_embed(x)

        # === 2. 添加 CLS Token ===
        # cls_token: [1, 1, 768] → [B, 1, 768]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # 拼接: [B, 1, 768] + [B, 196, 768] → [B, 197, 768]
        x = torch.cat([cls_tokens, x], dim=1)

        # === 3. 添加 Position Embedding ===
        # [B, 197, 768] + [1, 197, 768] → [B, 197, 768]
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # === 4. Transformer Blocks ===
        for block in self.blocks:
            x = block(x)

        # === 5. Layer Norm ===
        x = self.norm(x)

        # === 6. 提取 CLS Token 的输出 ===
        # [B, 197, 768] → [B, 768]
        cls_output = x[:, 0]

        return cls_output

    def forward(self, x):
        """
        完整前向传播
        x: [B, 3, 224, 224]
        return: [B, num_classes]
        """
        # 提取特征
        features = self.forward_features(x)  # [B, 768]

        # 分类
        logits = self.head(features)  # [B, num_classes]

        return logits


# ==================== 预定义配置 ====================

def vit_tiny_patch16_224(**kwargs):
    """ViT-Tiny: 5.7M params"""
    model = ViT(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        **kwargs
    )
    return model


def vit_small_patch16_224(**kwargs):
    """ViT-Small: 22M params"""
    model = ViT(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        **kwargs
    )
    return model


def vit_base_patch16_224(**kwargs):
    """ViT-Base: 86M params"""
    model = ViT(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        **kwargs
    )
    return model


def vit_large_patch16_224(**kwargs):
    """ViT-Large: 307M params"""
    model = ViT(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs
    )
    return model


def vit_huge_patch14_224(**kwargs):
    """ViT-Huge: 632M params"""
    model = ViT(
        img_size=224,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs
    )
    return model


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 创建模型
    model = vit_base_patch16_224(num_classes=1000)

    print("=" * 60)
    print("ViT-Base/16 模型信息")
    print("=" * 60)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")

    # 测试前向传播
    print("\n" + "=" * 60)
    print("前向传播测试")
    print("=" * 60)

    x = torch.randn(2, 3, 224, 224)
    print(f"输入: {x.shape}")

    with torch.no_grad():
        # 提取特征
        features = model.forward_features(x)
        print(f"特征: {features.shape}")

        # 分类
        logits = model(x)
        print(f"输出: {logits.shape}")

        # 预测类别
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        print(f"预测类别: {preds}")

    print("\n" + "=" * 60)
    print("模型结构 (主要组件)")
    print("=" * 60)
    print(model)