import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    """
    Dueling DQN 网络

    Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]

    共享主干提取特征，然后分为两个流：
      Value  流：估计状态价值 V(s)          → 标量
      Advantage 流：估计动作优势 A(s,a)    → act_dim 维向量
    """

    def __init__(self, obs_dim: int = 8, act_dim: int = 4, hidden: int = 256):
        super().__init__()

        # 共享特征提取
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Value 流：V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Advantage 流：A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.feature(obs)
        # 状态本身值多少 
        v = self.value_stream(h)                          # [B, 1]
        # 这个动作比其他动作好多少
        a = self.advantage_stream(h)                      # [B, act_dim]
        q = v + a - a.mean(dim=-1, keepdim=True)          # [B, act_dim]
        return q
