import torch
from torch import nn
import math
from config import T

# 正弦位置编码
# t ∈ [0, 1000] 标量       half_dim=128 [1.0, 0.930, 0.865, ..., 0.0001]  shape: (128,)  emb = log(10000) / 127 = 0.0725 ---> emb = exp(arange(128, device) * -0.0725)
#          ↓
# [128个高频到低频] 调制      shape [4, 1] × [1, 128] = [4, 128]     emb = [B][:, None] * emb[None, :] → [4, 128]
#          ↓
# 每个频率都计算 sin 和 cos   最后：sin + cos拼接 sin_emb: [4, 128] cos_emb: [4, 128] cat():   [4, 256]
#          ↓
# 拼接为 256 维向量
#  freq = torch.exp(-math.log(max_period) * torch.arange(0, half_dim) / half_dim)
#
class TimestepEmbedding(nn.Module):
    """时间步编码，使用sinusoidal position encoding"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        """
        timesteps: [B]
        return: [B, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        #  [0,1,2....half_emb]*(log(10000)/half_dim)
        #  embt = half_embxt
        #  concat = torch.cat((embtxsin,embtxcos),dim=-1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

if __name__ == '__main__':
    time_emb = TimestepEmbedding(16)
    t = torch.randint(0, T, (2,))  # 随机2个图片的t时间步
    embs = time_emb(t)
    print(embs)