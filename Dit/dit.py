import torch
import torch.nn as nn

from dit_block import DiTBlock, AdaLN
from label_emb import LabelEmbedding
from time_emb import TimestepEmbedding

# layer 28 , dim 1152, heads 16
# layer 24 , dim 1024, heads 16
# layer 12 , dim 768, heads 12
#  输入加了噪声的图片，时刻t, 提示词
# DiT的核心创新点——用纯Transformer架构完全替代了U-Net。
class DiT(nn.Module):
    def __init__(self,
                 img_size=28, # 输入图像尺寸 (MNIST: 28×28)
                 patch_size=4, # 将图像切分成 4×4 的patch
                 in_channels=1, # 输入通道数 (灰度图)
                 hidden_dim=256,
                 depth=6, # 层数
                 num_heads=8, # 多头注意力的头数
                 mlp_ratio=4.0,
                 num_classes=10,  # 新增：类别数量
                 class_dropout_prob=0.1  # 新增：Classifier-Free Guidance的dropout概率
        ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.num_patches = (img_size // patch_size) ** 2 # 计算patch数量 把 28×28 的图像切成 7×7 = 49 个 patch

        #  patchify 通过卷积,patch_size 是切成块的个数
        #  VIT 也是用的同样的方式做 patchify
        #  例如：图片 224x224x3, patch_size=16, 16wx16hx3channel--> 每块1*1*768 一共14x14x768 块，然后在拉平成 196x768x1
        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,  #输出的维度，卷积核的数量 768
            kernel_size=patch_size,
            stride=patch_size
        )

        # 位置编码 (可学习) 让模型知道每个 patch 的位置信息
        # patch1 + pos_embed[0] = [0.1+p1, 0.2+p2, 0.3+p3]  # 带位置0的标记
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        # time embedding 位置编码
        self.time_embed = nn.Sequential(
            TimestepEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # num_classes + 1 是因为要保留一个位置给"无条件"生成
        self.label_embed = LabelEmbedding(num_classes + 1, hidden_dim)

        # 组合时间步和标签的MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        #Dit Blocks ,其中 gamma 和 beta 通过线性层生成
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # layer norm
        self.final_adaLN = AdaLN(hidden_dim, hidden_dim)

        # linear and reshape， 同patchify相同形状
        self.final_layer = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        """初始化权重"""
        # 位置编码用截断正态分布初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 其他层用xavier初始化
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def unpatchify(self, x):
        """
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
        x: [B, C, H, W] 噪声图像
        t: [B] 时间步
        y: [B] 类别标签 (None表示无条件生成)
        """
        B = x.shape[0]

        # 1. Patch embedding
        x = self.patch_embed(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        # 2. 加入位置编码
        x = x + self.pos_embed

        # 3. 时间步编码
        t_emb = self.time_embed(t)  # [B, D]】

        # 3. 类别编码 (Classifier-Free Guidance)
        if y is None:
            # 无条件生成:使用最后一个索引作为"空"标签
            y = torch.full((B,), self.num_classes,device=x.device, dtype=torch.long)

        if self.training:
            dropout_mask = torch.rand(B, device=x.device) < self.class_dropout_prob
            y = torch.where(dropout_mask, self.num_classes, y)
        y_emb = self.label_embed(y)  # [B, D]
        # 4. 融合条件
        c = self.cond_mlp(torch.cat([t_emb, y_emb], dim=1))  # [B, D]
        # 4. 通过DiT blocks
        for block in self.blocks:
            x = block(x, c)  # 传入融合后的条件

        # 5. 最终归一化和投影
        x = self.final_adaLN(x, c)#和归一化有什么区别
        x = self.final_layer(x)  # [B, N, P*P*C]

        # 6. 还原成图像
        x = self.unpatchify(x)  # [B, C, H, W]

        return x


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DiT(
        img_size=28,
        patch_size=4,
        in_channels=1,
        hidden_dim=256,
        depth=6,
        num_heads=8
    ).to(device)

    # 打印模型信息
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(4, 1, 28, 28).to(device)
    t = torch.randint(0, 1000, (4,)).to(device)

    with torch.no_grad():
        output = model(x, t)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("模型测试通过!")