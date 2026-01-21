import torch
import torch.nn as nn

class AdaLN(nn.Module):
    """自适应层归一化 - 核心创新点"""
    # 条件向量: [time_embed(500), label_embed(3)]
    #          ↓ MLP
    # 生成参数: γ = [1.2, 0.8, 1.5, ...], β = [0.1, -0.2, 0.3, ...]
    #          ↓ 广播到所有token
    # 影响:
    #   token_1: 原特征 [a1, b1, c1] → 调制后 [1.2*a1+0.1, 0.8*b1-0.2, 1.5*c1+0.3]
    #   token_2: 原特征 [a2, b2, c2] → 调制后 [1.2*a2+0.1, 0.8*b2-0.2, 1.5*c2+0.3]
    #   ...
    #   token_N: 原特征 [aN, bN, cN] → 调制后 [1.2*aN+0.1, 0.8*bN-0.2, 1.5*cN+0.3]
    #  所有token受到相同的调制 , 所有像素位置得到相同的"指导信号"
    #  "整体地"告诉网络: "现在噪声很大,需要大幅调整,目标是数字3"
    #  Cross Attention 的表达能力更好,参数也更大
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        # 从条件生成 gamma 和 beta
        self.linear = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, cond):
        """
        x: [B, N, D] 输入特征
        cond: [B, cond_dim] 条件embedding (时间步)
        """
        # 先归一化
        x_norm = self.norm(x)

        # 生成调制参数
        scale_shift = self.linear(cond)  # [B, 2*D]
        gamma, beta = scale_shift.chunk(2, dim=-1)  # 各 [B, D]

        # 调制: gamma * norm(x) + beta
        return gamma.unsqueeze(1) * x_norm + beta.unsqueeze(1)

class DiTBlock(nn.Module):
    """DiT的基本模块"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # AdaLN for attention
        self.adaLN1 = AdaLN(dim, dim)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # AdaLN for MLP
        self.adaLN2 = AdaLN(dim, dim)

        # MLP
        #  供nn.Linear 等需要整数尺寸的层使用，mlp_ratio 隐藏层维度的扩展倍数
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x, cond):
        """
        x: [B, N, D] 输入tokens
        cond: [B, D] 条件embedding
        """
        # Attention with residual
        x_norm = self.adaLN1(x, cond)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x_norm = self.adaLN2(x, cond)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x
