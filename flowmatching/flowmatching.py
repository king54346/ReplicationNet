import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间编码 - 优化版"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2

        # 预计算频率，避免每次forward重复计算
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        """
        t: [B] 时间步 (0到1之间)
        return: [B, dim] 时间编码
        """
        # 计算编码
        args = t[:, None].float() * self.freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class SelfAttention(nn.Module):
    """多头自注意力模块"""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, "channels必须能被num_heads整除"

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Group Normalization
        self.norm = nn.GroupNorm(8, channels)

        # QKV投影
        self.qkv = nn.Conv2d(channels, channels * 3, 1)

        # 输出投影
        self.proj_out = nn.Conv2d(channels, channels, 1)

        # 零初始化输出投影，使注意力初始时为恒等映射
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, C, H, W]
        """
        B, C, H, W = x.shape
        residual = x

        # 归一化
        x = self.norm(x)

        # 计算QKV
        qkv = self.qkv(x)  # [B, 3*C, H, W]
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, num_heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, H*W, H*W]
        attn = F.softmax(attn, dim=-1)

        # 应用注意力
        out = torch.matmul(attn, v)  # [B, num_heads, H*W, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        # 输出投影
        out = self.proj_out(out)

        return residual + out


class ResBlock(nn.Module):
    """残差块 with AdaGN (Adaptive Group Normalization)"""

    def __init__(self, channels, time_dim, dropout=0.1, out_channels=None):
        super().__init__()
        out_channels = out_channels or channels

        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, out_channels, 3, padding=1)

        # 时间投影层 - AdaGN
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2)
        )

        # 零初始化，使AdaGN初始时近似恒等变换
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # 残差连接
        if channels != out_channels:
            self.shortcut = nn.Conv2d(channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        """
        x: [B, C, H, W]
        t_emb: [B, time_dim]
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # 时间信息注入 - AdaGN
        t = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = t.chunk(2, dim=1)
        h = h * (1 + scale) + shift

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.shortcut(x) + h


class DownBlock(nn.Module):
    """下采样块"""

    def __init__(self, in_channels, out_channels, time_dim, num_res_blocks=2,
                 dropout=0.1, use_attention=False, num_heads=4):
        super().__init__()

        # ResBlocks
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels if i == 0 else out_channels,
                time_dim,
                dropout,
                out_channels
            )
            for i in range(num_res_blocks)
        ])

        # 可选的注意力层
        if use_attention:
            self.attentions = nn.ModuleList([
                SelfAttention(out_channels, num_heads)
                for _ in range(num_res_blocks)
            ])
        else:
            self.attentions = None

        # 下采样（stride=2 精确减半）
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        """
        x: [B, C_in, H, W]
        t_emb: [B, time_dim]
        return:
            x_down: [B, C_out, H//2, W//2] 下采样后的特征
            x_skip: [B, C_out, H, W] 用于跳跃连接的特征
        """
        h = x
        for i, res_block in enumerate(self.res_blocks):
            h = res_block(h, t_emb)
            if self.attentions is not None:
                h = self.attentions[i](h)

        x_skip = h
        x_down = self.downsample(h)

        return x_down, x_skip


class UpBlock(nn.Module):
    """上采样块"""

    def __init__(self, in_channels, out_channels, time_dim, num_res_blocks=2,
                 dropout=0.1, use_attention=False, num_heads=4):
        super().__init__()

        # 上采样（使用插值 + 卷积，更稳定且尺寸精确）
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 尺寸精确翻倍
            nn.Conv2d(in_channels, out_channels, 3, padding=1)  # 平滑特征
        )

        # ResBlocks
        # 第一个ResBlock: 输入是上采样特征(out_channels) + 跳跃连接(out_channels) = out_channels * 2
        # 后续ResBlock: 输入和输出都是out_channels
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            in_ch = out_channels * 2 if i == 0 else out_channels
            self.res_blocks.append(
                ResBlock(in_ch, time_dim, dropout, out_channels)
            )

        # 可选的注意力层
        if use_attention:
            self.attentions = nn.ModuleList([
                SelfAttention(out_channels, num_heads)
                for _ in range(num_res_blocks)
            ])
        else:
            self.attentions = None

    def forward(self, x, x_skip, t_emb):
        """
        x: [B, C_in, H, W] 来自上一层的特征
        x_skip: [B, C_out, H_skip, W_skip] 来自编码器的跳跃连接
        t_emb: [B, time_dim]
        return: [B, C_out, H_skip, W_skip]
        """
        # 上采样
        h = self.upsample(x)
        
        # 动态调整尺寸以匹配跳跃连接
        # 这对于奇数尺寸很重要（例如 7x7）
        if h.shape[2:] != x_skip.shape[2:]:
            h = F.interpolate(h, size=x_skip.shape[2:], mode='nearest')
        
        # 拼接跳跃连接
        h = torch.cat([h, x_skip], dim=1)
        
        # ResBlocks + Attention
        for i, res_block in enumerate(self.res_blocks):
            h = res_block(h, t_emb)
            if self.attentions is not None:
                h = self.attentions[i](h)
        
        return h


class MidBlock(nn.Module):
    """中间块 (Bottleneck)"""

    def __init__(self, channels, time_dim, num_res_blocks=3, dropout=0.1, num_heads=4):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for i in range(num_res_blocks):
            self.res_blocks.append(ResBlock(channels, time_dim, dropout))
            # 在每个ResBlock后添加注意力（除了最后一个）
            if i < num_res_blocks - 1:
                self.attentions.append(SelfAttention(channels, num_heads))

    def forward(self, x, t_emb):
        """
        x: [B, C, H, W]
        t_emb: [B, time_dim]
        return: [B, C, H, W]
        """
        h = x
        for i, res_block in enumerate(self.res_blocks):
            h = res_block(h, t_emb)
            if i < len(self.attentions):
                h = self.attentions[i](h)
        return h


class FlowMatchingModel(nn.Module):
    def __init__(
            self,
            img_channels=1,
            base_channels=64,
            channel_mult=(1, 2, 4),
            time_dim=256,
            num_classes=10,
            num_res_blocks=2,
            dropout=0.1,
            use_attention=(False, True, True),  # 在哪些层级使用注意力
            num_heads=4
    ):
        super().__init__()

        # 时间编码
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)

        # 类别编码 (num_classes + 1 for unconditional)
        self.label_embedding = nn.Embedding(num_classes + 1, time_dim)

        # 时间+类别 embedding 融合 MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # 初始卷积
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        # 构建编码器
        self.down_blocks = nn.ModuleList()
        channels_list = [base_channels * mult for mult in channel_mult]
        in_ch = base_channels
        # channel multi 1 ，2， 4
        # channel count 64，128，256
        # 图片尺寸 28，14，7
        for i, out_ch in enumerate(channels_list):
            use_attn = use_attention[i] if i < len(use_attention) else False
            self.down_blocks.append(
                DownBlock(
                    in_ch, out_ch, time_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    use_attention=use_attn,
                    num_heads=num_heads
                )
            )
            in_ch = out_ch

        # 中间块
        self.mid_block = MidBlock(
            channels_list[-1], time_dim,
            num_res_blocks=3,
            dropout=dropout,
            num_heads=num_heads
        )

        # 构建解码器
        # 解码器需要与编码器严格对称
        # channels_list = [64, 128, 256]
        # skip连接:    [64@28, 128@14, 256@7]
        # 
        # Up1: 256 -> 256 (upsample) + 256 (skip) -> 256 (output)
        # Up2: 256 -> 128 (upsample) + 128 (skip) -> 128 (output)
        # Up3: 128 -> 64 (upsample) + 64 (skip) -> 64 (output)
        
        self.up_blocks = nn.ModuleList()
        
        # 倒序遍历编码器的通道配置
        for i in range(len(channels_list) - 1, -1, -1):
            # 当前层的跳跃连接通道数
            skip_ch = channels_list[i]
            
            # 输入通道数：来自上一个UpBlock的输出（或MidBlock）
            if i == len(channels_list) - 1:
                # 第一个UpBlock，输入来自MidBlock
                in_ch = channels_list[i]
            else:
                # 后续UpBlock，输入来自上一个UpBlock的输出
                in_ch = channels_list[i + 1]
            
            # 输出通道数：应该等于跳跃连接的通道数
            out_ch = skip_ch
            
            # 注意力配置
            use_attn = use_attention[i] if i < len(use_attention) else False

            self.up_blocks.append(
                UpBlock(
                    in_ch, out_ch, time_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    use_attention=use_attn,
                    num_heads=num_heads
                )
            )

        # 输出层 - 使用合理的初始化
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, 3, padding=1)
        )

        # 使用小方差初始化，而非完全零初始化
        # 这样既保持初始输出较小，又不会完全阻止梯度流动
        nn.init.normal_(self.final_conv[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.final_conv[-1].bias)

    def forward(self, x, t, y=None):
        """
        x: [B, C, H, W] 当前状态
        t: [B] 时间步 (0到1)
        y: [B] 类别标签 (可选)
        return: [B, C, H, W] 预测的速度场
        """
        # 时间编码
        t_emb = self.time_embedding(t)

        # 类别编码 (如果没有提供，使用无条件类别)
        if y is None:
            y = torch.full((x.size(0),), self.label_embedding.num_embeddings - 1,
                           device=x.device, dtype=torch.long)
        y_emb = self.label_embedding(y)

        # 组合条件并通过MLP
        cond = t_emb + y_emb
        cond = self.cond_mlp(cond)

        # 初始卷积
        h = self.init_conv(x)

        # 编码器 - 保存跳跃连接
        skip_connections = []
        for down_block in self.down_blocks:
            h, h_skip = down_block(h, cond)
            skip_connections.append(h_skip)

        # 中间块
        h = self.mid_block(h, cond)

        # 解码器 - 使用跳跃连接
        for up_block in self.up_blocks:
            h_skip = skip_connections.pop()
            h = up_block(h, h_skip, cond)

        # 输出
        v = self.final_conv(h)

        return v


def check_model_initialization():
    """检查模型初始化是否正常"""
    print("=" * 80)
    print("模型初始化检查")
    print("=" * 80)

    model = FlowMatchingModel(
        img_channels=1,
        base_channels=64,
        channel_mult=(1, 2, 4),
        time_dim=256,
        num_classes=10,
        num_res_blocks=2,
        use_attention=(False, True, True),
        num_heads=4
    ).cuda()

    # 打印模型结构
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"  总参数: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")

    # 检查 freqs 是否是 buffer
    print(f"\n时间编码优化:")
    print(f"  freqs 已注册为 buffer: {'freqs' in dict(model.time_embedding.named_buffers())}")

    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, 1, 32, 32).cuda()
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

    # 检查final_conv初始化
    final_weight = model.final_conv[-1].weight
    print(f"\nfinal_conv 初始化:")
    print(f"  weight: mean={final_weight.mean():.6f}, std={final_weight.std():.6f}")
    print(f"  weight: min={final_weight.min():.6f}, max={final_weight.max():.6f}")

    # 检查是否正常
    if v.abs().max() < 1e-6:
        print(f"\n⚠️  警告: 输出接近全零! 这会导致生成黑图")
        print(f"  可能原因:")
        print(f"    1. 输出层初始化过小")
        print(f"    2. 梯度消失")
        return False
    elif v.abs().mean() < 0.001:
        print(f"\n⚠️  警告: 输出幅度很小,但可以接受")
        print(f"  这是正常的小初始化，训练后会增大")
        return True
    else:
        print(f"\n✓ 模型输出正常")
        return True


if __name__ == '__main__':
    check_model_initialization()