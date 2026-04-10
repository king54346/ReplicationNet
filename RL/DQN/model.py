"""
DQN 模型定义
============
包含:
  - QNetwork        : Q 值网络 (输入状态 → 输出每个动作的 Q 值)
  - ReplayBuffer    : 经验回放缓冲区
  - DQNAgent        : 整合网络、Buffer、训练逻辑的 Agent

核心思想:
  Q(s, a) 表示"在状态 s 执行动作 a 后的期望累计折扣奖励"
  Bellman 方程: Q(s,a) = r + γ * max_a' Q(s', a')
  用神经网络拟合该方程，结合 Experience Replay + Target Network 稳定训练。
"""

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────
# 1. Q 网络
# ─────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    全连接 Q 网络。

    输入 : 状态向量 s  (shape: [batch, state_dim])
    输出 : 每个动作的 Q 值  (shape: [batch, action_dim])

    结构:
        Linear(state_dim → hidden) → ReLU
        Linear(hidden    → hidden) → ReLU
        Linear(hidden    → action_dim)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────
# 2. 经验回放缓冲区
# ─────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    循环队列，存储 (s, a, r, s', done) 转移元组。

    作用:
      - 解耦时序相关性：随机采样使训练样本近似 i.i.d.
      - 提升样本利用率：同一条经验可被多次采样

    参数:
      capacity : 队列最大长度，超出时自动丢弃最旧的样本
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存入一条转移经验。"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        随机采样 batch_size 条经验。

        返回:
          (states, actions, rewards, next_states, dones)
          均为 numpy array，dtype 分别为 float32 / int64 / float32 / float32 / float32
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────
# 3. DQN Agent
# ─────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    DQN Agent，封装以下模块:
      - Online Network  (q_net)     : 实时更新，用于选动作和计算当前 Q
      - Target Network  (target_net): 定期同步，用于生成稳定的 TD 目标
      - ReplayBuffer                : 存储与采样历史经验
      - ε-greedy 策略               : 平衡探索与利用

    训练目标 (TD 误差):
      loss = SmoothL1( Q_online(s,a),  r + γ * max_a' Q_target(s',a') * (1-done) )

    参数说明:
      state_dim          : 状态向量维度
      action_dim         : 离散动作数量
      lr                 : Adam 学习率
      gamma              : 折扣因子 γ
      epsilon_start      : ε 初始值（纯随机探索）
      epsilon_end        : ε 最小值（保留少量随机性）
      epsilon_decay      : 每回合乘以该系数衰减 ε
      buffer_size        : ReplayBuffer 容量
      batch_size         : 每次训练采样数量
      target_update_freq : 每隔多少回合同步一次 Target Network
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 10,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online Network：每步梯度更新
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        # Target Network：定期硬拷贝，充当固定靶子
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    # ── ε-greedy 动作选择 ──────────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        """
        以概率 ε 随机选动作（探索），否则选 Q 值最大的动作（利用）。
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax(dim=1).item()

    # ── 存储经验 ────────────────────────────────────────────────────
    def store(self, state, action, reward, next_state, done):
        """将一条转移经验压入 ReplayBuffer。"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ── 训练一步 ────────────────────────────────────────────────────
    def train(self):
        """
        从 Buffer 随机采样，计算 TD 误差并更新 Online Network。

        流程:
          1. 采样 batch
          2. 计算 current_q  = Q_online(s, a)         (gather 实际执行的动作)
          3. 计算 target_q   = r + γ * max Q_target(s')  (done=True 时截断)
          4. Huber Loss + 梯度裁剪 + Adam 更新

        返回:
          loss (float) 或 None（Buffer 不足时跳过）
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 当前 Q 值：只取实际执行那个动作的 Q 值
        current_q = self.q_net(states_t).gather(1, actions_t)

        # TD 目标：用 Target Network 估计下一步价值（不参与梯度）
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q   = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        # Huber Loss（对异常值比 MSE 更鲁棒）
        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)  # 梯度裁剪
        self.optimizer.step()

        self.train_step += 1
        return loss.item()

    # ── 同步 Target Network ─────────────────────────────────────────
    def sync_target(self):
        """将 Online Network 参数硬拷贝到 Target Network。"""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ── 衰减 ε ─────────────────────────────────────────────────────
    def decay_epsilon(self):
        """每回合结束后调用，按指数衰减探索率。"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ── 保存 / 加载权重 ─────────────────────────────────────────────
    def save(self, path: str):
        """保存 Online Network 权重到文件。"""
        torch.save(self.q_net.state_dict(), path)
        print(f"[Model] 权重已保存 → {path}")

    def load(self, path: str):
        """从文件加载权重到 Online Network（同时同步 Target Network）。"""
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.q_net.eval()
        print(f"[Model] 权重已加载 ← {path}")

