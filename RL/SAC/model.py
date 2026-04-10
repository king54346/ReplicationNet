"""
SAC (Soft Actor-Critic) 模型定义
==================================
论文: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
      with a Stochastic Actor (Haarnoja et al., 2018)

核心思想:
  传统 RL 目标:   max  E[ Σ r(s,a) ]
  SAC 目标:       max  E[ Σ r(s,a) + α * H(π(·|s)) ]
                                         ↑
                               策略熵 (Entropy)，鼓励探索

  同时最大化累计奖励 和 策略熵，两个好处:
    1. 防止策略过早收敛到次优解（保持探索）
    2. 遇到多个同等好的动作时，倾向于均匀选择（更鲁棒）

与 DQN 的关键区别:
  DQN         离散动作，确定性策略，无熵正则
  SAC         连续动作，随机策略（高斯），自动调节熵系数 α

网络结构:
  Actor (策略网络):
    s → [μ(s), log_σ(s)]  → 重参数化采样  → tanh → a ∈ [-1, 1]
                               a = tanh(μ + σ * ε),  ε ~ N(0,1)
    用 tanh 把动作压缩到 [-1,1]，同时需修正对数概率（log_prob）

  Critic (Q 网络，两个):
    (s, a) → Q 值
    用两个 Q 网络取 min，缓解 Q 值高估问题（Double Q-trick）

  自动熵调节:
    α 不固定，训练时通过梯度自动调整，使熵接近目标熵 H_target = -dim(A)

模块:
  - GaussianActor    : 输出高斯分布均值和方差，支持重参数化采样
  - QNetwork         : (state, action) → Q 值
  - ReplayBuffer     : 经验回放（与 DQN 相同）
  - SACAgent         : 整合 Actor + 2×Critic + Target Critic + 自动 α
"""

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


LOG_STD_MAX =  2    # log σ 的上界（方差不能太大）
LOG_STD_MIN = -20   # log σ 的下界（方差不能太小导致梯度消失）


# ─────────────────────────────────────────────────────────────────
# 1. 高斯 Actor 网络（随机策略）
# ─────────────────────────────────────────────────────────────────
class GaussianActor(nn.Module):
    """
    输出连续动作的随机策略：π(a|s) = N(μ(s), σ(s))

    重参数化技巧 (Reparameterization Trick):
      直接采样 a ~ π(a|s) 无法反向传播（随机节点阻断梯度）
      改为: ξ ~ N(0,1),  a_raw = μ + σ * ξ
      梯度可以流过 μ 和 σ，ξ 只是常数噪声

    tanh 压缩:
      a = tanh(a_raw) ∈ (-1, 1)
      再乘以 action_scale 映射到实际动作范围

    log_prob 修正（tanh Jacobian）:
      原始高斯的 log_prob 要减去 tanh 变换的 log Jacobian
      log π(a|s) = log N(ξ; 0,1) - Σ log(1 - tanh²(a_raw))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(hidden, action_dim)
        self.log_std_layer = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor):
        """
        给定状态 s，返回动作均值 μ 和对数标准差 log_σ。
        """
        feat    = self.shared(state)
        mean    = self.mean_layer(feat)
        log_std = self.log_std_layer(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        """
        重参数化采样，返回 (action, log_prob, mean)。

        action   : tanh 压缩后的动作 ∈ (-1, 1)^action_dim
        log_prob : 修正后的对数概率，用于计算策略熵损失
        mean     : 确定性动作（评估时使用）

        Returns:
          action   (B, action_dim)
          log_prob (B, 1)
          mean     (B, action_dim)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # 重参数化: a_raw = μ + σ * ε,  ε ~ N(0,1)
        normal  = torch.distributions.Normal(mean, std)
        a_raw   = normal.rsample()          # rsample 支持梯度反传

        # tanh 压缩
        action  = torch.tanh(a_raw)

        # log_prob 修正：减去 tanh Jacobian 的 log
        # log_prob = Σ[ log N(ξ) - log(1 - tanh²(a_raw)) ]
        log_prob = normal.log_prob(a_raw) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)   # (B, 1)

        return action, log_prob, torch.tanh(mean)


# ─────────────────────────────────────────────────────────────────
# 2. Q 网络（Critic）
# ─────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    连续动作版 Q 网络：输入 (state, action)，输出 Q(s, a) 标量。

    与 DQN 的区别：
      DQN:  state → Q(s, a0), Q(s, a1), ...    # 离散动作
      SAC:  (state, action) → Q(s,a)            # 连续动作，action 直接拼接进去
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        输入: state (B, state_dim), action (B, action_dim)
        输出: Q 值 (B, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────────────────────────
# 3. 经验回放缓冲区
# ─────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    存储 (s, a, r, s', done) 转移元组，随机采样打破时序相关性。
    与 DQN 完全一致，SAC 同样是 Off-Policy 算法，可重用历史经验。
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.float32),
            np.array(rewards,     dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────
# 4. SAC Agent
# ─────────────────────────────────────────────────────────────────
class SACAgent:
    """
    Soft Actor-Critic Agent。

    包含的网络:
      actor          : GaussianActor（策略网络）
      q1, q2         : 两个 Online Critic（Q 网络）
      q1_tgt, q2_tgt : 两个 Target Critic（软更新，τ 步长）

    三个优化目标（分别独立更新）:
      ① Critic 损失:
           target_q = r + γ * (min(Q1_tgt, Q2_tgt)(s', ã') - α * log π(ã'|s'))
           loss_q   = MSE(Q1(s,a), target_q) + MSE(Q2(s,a), target_q)

      ② Actor 损失 (最大化期望 Q - 熵):
           loss_π = E_s[ α * log π(ã|s) - min(Q1(s,ã), Q2(s,ã)) ]
           其中 ã 是从当前策略采样（重参数化，可反传梯度）

      ③ 温度 α 损失 (自动调节熵):
           H_target = -dim(A)     （目标熵，超参数）
           loss_α   = E[ -α * (log π(ã|s) + H_target) ]
           α 增大 → 更鼓励探索（熵太低时）
           α 减小 → 更专注利用（熵太高时）

    软更新 (Soft Update / Polyak Averaging):
      θ_tgt ← τ * θ + (1-τ) * θ_tgt
      比 DQN 的硬更新（每 N 步直接复制）更平滑，通常 τ = 0.005

    参数说明:
      state_dim    : 状态空间维度
      action_dim   : 动作空间维度
      action_scale : 动作范围上界（用于 rescale tanh 输出）
      lr           : 所有网络的学习率
      gamma        : 折扣因子
      tau          : 软更新系数
      alpha_init   : 初始温度系数
      auto_alpha   : 是否自动调节 α
      buffer_size  : ReplayBuffer 容量
      batch_size   : 训练 batch 大小
    """

    def __init__(
        self,
        state_dim    : int,
        action_dim   : int,
        action_scale : float = 1.0,
        lr           : float = 3e-4,
        gamma        : float = 0.99,
        tau          : float = 0.005,
        alpha_init   : float = 0.2,
        auto_alpha   : bool  = True,
        buffer_size  : int   = 100_000,
        batch_size   : int   = 256,
    ):
        self.gamma       = gamma
        self.tau         = tau
        self.batch_size  = batch_size
        self.auto_alpha  = auto_alpha
        self.action_scale = action_scale

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Actor ──────────────────────────────────────────────────
        self.actor = GaussianActor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # ── Critic（Double Q-trick） ────────────────────────────────
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_tgt = QNetwork(state_dim, action_dim).to(self.device)
        self.q2_tgt = QNetwork(state_dim, action_dim).to(self.device)
        # 初始化 Target = Online
        self.q1_tgt.load_state_dict(self.q1.state_dict())
        self.q2_tgt.load_state_dict(self.q2.state_dict())
        self.q1_tgt.eval()
        self.q2_tgt.eval()
        self.critic_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

        # ── 温度系数 α ─────────────────────────────────────────────
        if auto_alpha:
            # 目标熵：连续动作空间通常取 -dim(A)
            self.target_entropy = -float(action_dim)
            # log_α 作为可训练参数（保证 α > 0）
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha_init

        # ── ReplayBuffer ───────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(buffer_size)

    # ── 动作选择 ──────────────────────────────────────────────────
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        采样动作。
          训练时: deterministic=False，从随机策略采样（保留探索）
          评估时: deterministic=True，用均值（确定性最优动作）

        返回 numpy array，乘以 action_scale 映射到实际动作范围。
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                _, _, mean = self.actor.sample(state_t)
                action = mean
            else:
                action, _, _ = self.actor.sample(state_t)
        return (action.cpu().numpy()[0] * self.action_scale)

    # ── 存储经验 ──────────────────────────────────────────────────
    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ── 训练一步 ──────────────────────────────────────────────────
    def train(self):
        """
        从 Buffer 采样，依次更新 Critic → Actor → α。

        返回:
          dict: { "critic_loss", "actor_loss", "alpha_loss", "alpha" }
          或 None（Buffer 不足时跳过）
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.FloatTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        s_ = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)

        # ① 更新 Critic ─────────────────────────────────────────
        with torch.no_grad():
            # 从当前策略对 s' 采样动作（带熵）
            a_next, log_prob_next, _ = self.actor.sample(s_)
            a_next_scaled = a_next * self.action_scale

            # Target Q（取两个 Critic 的 min，缓解高估）
            q1_next = self.q1_tgt(s_, a_next_scaled)
            q2_next = self.q2_tgt(s_, a_next_scaled)
            min_q_next = torch.min(q1_next, q2_next)

            # 软贝尔曼目标：r + γ * (min_Q - α * log π)
            # 减去 α * log π 即最大化熵的体现
            target_q = r + self.gamma * (1 - d) * (min_q_next - self.alpha * log_prob_next)

        q1_val = self.q1(s, a)
        q2_val = self.q2(s, a)
        critic_loss = F.mse_loss(q1_val, target_q) + F.mse_loss(q2_val, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ② 更新 Actor ──────────────────────────────────────────
        # 冻结 Critic 参数，只更新 Actor
        a_new, log_prob, _ = self.actor.sample(s)
        a_new_scaled = a_new * self.action_scale

        q1_new = self.q1(s, a_new_scaled)
        q2_new = self.q2(s, a_new_scaled)
        min_q_new = torch.min(q1_new, q2_new)

        # 最大化 (Q - α * log π) ↔ 最小化 (α * log π - Q)
        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ③ 更新温度 α（自动熵调节）──────────────────────────────
        alpha_loss_val = 0.0
        if self.auto_alpha:
            # 目标：让熵接近 target_entropy
            # loss_α = -α * (log π(ã|s) + H_target)
            #          当 log π 太大（熵太低）→ loss_α > 0 → α 增大 → 更鼓励探索
            #          当 log π 太小（熵太高）→ loss_α < 0 → α 减小 → 更专注利用
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_val = alpha_loss.item()

        # ④ 软更新 Target Critic ─────────────────────────────────
        # θ_tgt ← τ * θ + (1-τ) * θ_tgt   （比 DQN 硬拷贝更平滑）
        for param, tgt_param in zip(self.q1.parameters(), self.q1_tgt.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)
        for param, tgt_param in zip(self.q2.parameters(), self.q2_tgt.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss" : actor_loss.item(),
            "alpha_loss" : alpha_loss_val,
            "alpha"      : self.alpha,
        }

    # ── 保存 / 加载权重 ───────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "actor"  : self.actor.state_dict(),
            "q1"     : self.q1.state_dict(),
            "q2"     : self.q2.state_dict(),
            "q1_tgt" : self.q1_tgt.state_dict(),
            "q2_tgt" : self.q2_tgt.state_dict(),
        }, path)
        print(f"[Model] 权重已保存 → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_tgt.load_state_dict(ckpt["q1_tgt"])
        self.q2_tgt.load_state_dict(ckpt["q2_tgt"])
        self.actor.eval()
        print(f"[Model] 权重已加载 ← {path}")

