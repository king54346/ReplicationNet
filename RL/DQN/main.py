"""
DQN (Deep Q-Network) - CartPole 示例
=====================================
论文: Playing Atari with Deep Reinforcement Learning (DeepMind, 2013)

核心技术:
  1. Experience Replay     - 打破样本相关性
  2. Target Network        - 稳定 Q 值目标
  3. Epsilon-Greedy        - 探索与利用的平衡

环境: CartPole-v1
  状态: [车位置, 车速度, 杆角度, 杆角速度] (4维)
  动作: {0: 向左推, 1: 向右推}
  目标: 保持杆子不倒，单回合最大 500 步
"""

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym



# target = 即时奖励 + gamma * 网络估计的未来价值
# Monte Carlo              DQN (TD)
#
# 等游戏结束？      必须等                  不用，走一步更新一步
# 目标值来源        真实累计奖励             即时奖励 + 网络估计
# 方差             高（长序列累积误差）      低
# 偏差             无偏                    有偏（估计值不准）
# 适合场景          回合短、奖励密集         回合长、稀疏奖励
# 训练：
# 训练目标 = r + γ * Q(s', a')
#                   ↑
#           这个 Q 是网络自己估的
#
# 但网络一开始什么都不知道，估的全是垃圾值
# 那不就是用垃圾训练垃圾？
# 关键洞察：靠近终点的状态先学会
# 游戏最后一步（即将结束）：
# s_倒下前 → a → r = -10,  done = True
#
# target = -10 + 0.99 * Q(s_终止) * (1 - done)
#                                     ↑
#                                done=1 这项直接为0！
#
# target = -10   ← 这个值是确定的！不依赖网络估计
# Q(s_危险, 向墙走) ≈ -10   ✅ 学会了
# 然后往前一步也变准了
# s_危险前一步:
#
# target = r + 0.99 * Q(s_危险, best_a)
#                         ↑
#                这个已经学准了！≈ -10
#
# target ≈ 1 + 0.99 * (-10) ≈ -8.9   ← 这个也变准了
# 价值传播（Value Propagation）
# 终止状态（确定）
#     ↓ 传播
# 倒下前1步（学准）
#     ↓ 传播
# 倒下前2步（学准）
#     ↓ 传播
# 倒下前3步（学准）
#     ...
# Q(s, a) ≈ 从 s 执行 a 后的真实期望回报
# 策略 π(s) = argmax Q(s,a) 变得接近最优
# 参数 θ 更新 → Q(s', a') 变了 → 目标变了 → 追着变化跑 → 像追一个移动的靶子，容易震荡发散
# Online Network (θ)        Target Network (θ⁻)
# ─────────────────         ──────────────────
# 每步都在更新              每 10 回合才同步一次
#
# 计算当前 Q(s,a)           计算目标 Q(s',a')
#      ↓                          ↓
#      └──────── loss ────────────┘
#                ↓
#           只更新左边
#
# 靶子（目标值）是固定的，追固定的靶子不会发散
#           初始化（随机）
#                 ↓
#     ┌─── 与环境交互，收集 (s,a,r,s',done) ───┐
#     │                                        │
#     │   从 Buffer 随机采样                    │
#     │           ↓                            │
#     │   target = r + γ * Q_target(s', a')    │
#     │           ↓                            │
#     │   loss = (Q_online(s,a) - target)²     │
#     │           ↓                            │
#     │   反向传播，更新 Online Network          │
#     │           ↓                            │
#     │   每 N 步：θ⁻ ← θ  (同步 Target Net)  │
#     └────────────────────────────────────────┘
#                 ↓ 重复数万次
#            Q 值收敛到真实价值
# 走第1步 → 存入Buffer
# 走第2步 → 存入Buffer
# ...
# 走第64步 → Buffer凑够了 → 开始第一次训练
# 走第65步 → 存入Buffer + 再训练一次
# 走第66步 → 存入Buffer + 再训练一次
#  隐式的 Bootstrap 过程 ： 价值从终止状态一层一层"渗透"回初始状态
# 时间轴:
#   [0, 1k步]    纯随机探索，buffer积累
#   [1k, 10k步]  开始训练，终止状态附近的Q率先收敛
#   [10k, 100k步] 价值逐步向前传播，中间状态Q值收敛
#   [100k+步]    初始状态Q值稳定，策略趋于最优



# ─────────────────────────────────────────
# 1. Q 网络 (Neural Network)
# 输入状态 s，输出每个动作的 Q(s,a)  Action-Value Function
# 我现在处于状态 s，如果执行动作 a，从长远来看能获得多少总奖励？
# Bellman 方程
# γ 折扣因子，未来的奖励要打折（今天的 1 块 > 明天的 1 块）
# 当前动作的价值 = 即时奖励 + 下一步最优价值的折扣
# DQN 就是用神经网络拟合这个方程，不断迭代直到收敛
# ─────────────────────────────────────────
class QNetwork(nn.Module):
    """
    输入: 状态向量 s
    输出: 每个动作的 Q 值 Q(s, a)
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


# ─────────────────────────────────────────
# 2. 经验回放缓冲区 (Replay Buffer)
# ─────────────────────────────────────────
class ReplayBuffer:
    """
    存储 (s, a, r, s', done) 元组
    随机采样打破时序相关性，提升训练稳定性
    """
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        # # 从 10000 条经历里，随机抽 64 条
        # # 这 64 条可能来自第 3 步、第 892 步、第 5431 步...
        # # 时间上完全不连续！这就打破了相关性
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────
# 3. DQN Agent
# ─────────────────────────────────────────
class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,          # 折扣因子
        epsilon_start: float = 1.0,   # 初始探索率
        epsilon_end: float = 0.05,    # 最小探索率
        epsilon_decay: float = 0.995, # 衰减系数
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 10, # 每 N 回合同步 Target Network
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

        # Online Network: 实时更新
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        # Target Network: 定期同步，用于生成稳定的 TD 目标
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    # ── 动作选择 (ε-greedy) ──────────────────
    def select_action(self, state: np.ndarray) -> int:
        """
        以 epsilon 概率随机探索，否则贪心选择最优动作
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax(dim=1).item()

    # ── 存储经验 ─────────────────────────────
    # 存储 (s,a,r,s′,done) 随机采样打破时序相关性
    # s    = [0.02, 0.01, -0.03, 0.02]   # 车位置/速度, 杆角度/角速度
    # a    = 1                            # 向右推
    # r    = 1.0                          # 没倒，得1分
    # s'   = [0.02, 0.21, -0.03, -0.27]  # 推完之后的新状态
    # done = False                        # 游戏没结束

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ── 训练一步 ─────────────────────────────
    # ┌─────────────────────────────────────────────────────┐
    # │                   训练全流程                          │
    # │                                                     │
    # │  环境交互                    Replay Buffer           │
    # │  ──────────                  ────────────           │
    # │                                                     │
    # │  s0 → a0 → r0, s1  ──push──► [exp0]                │
    # │  s1 → a1 → r1, s2  ──push──► [exp0, exp1]          │
    # │  s2 → a2 → r2, s3  ──push──► [exp0, exp1, exp2]    │
    # │  ...                         [exp0...exp9999]  满了 │
    # │                               最老的自动被挤出        │
    # │                                                     │
    # │             ↓ 每步都 random sample(64条)             │
    # │                                                     │
    # │         [exp88, exp3001, exp472, ...]  ← 随机抽取   │
    # │                    ↓                               │
    # │              计算 loss，更新网络                      │
    # └─────────────────────────────────────────────────────┘
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 当前 Q 值: Q(s, a)
        #  只取实际执行的那个动作的 Q 值
        current_q = self.q_net(states_t).gather(1, actions_t)

        # TD 目标: r + γ * max_a' Q_target(s', a')
        # 用 Target Network 计算右边（稳定的旧网络）
        # 用 Online Network 计算左边（正在训练的网络）

        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            #                  done=True 时这项为0，不加未来奖励
            target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        # Huber Loss (比 MSE 对异常值更鲁棒)
        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        self.train_step += 1
        return loss.item()

    # ── 同步 Target Network ───────────────────
    # 每 10 回合把 online network 的参数复制到 target network
    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ── 衰减 epsilon ──────────────────────────
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ─────────────────────────────────────────
# 4. 训练主循环
# ─────────────────────────────────────────
def train_dqn(num_episodes: int = 400, render: bool = False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    state_dim  = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2

    agent = DQNAgent(state_dim, action_dim)

    reward_history = []
    best_avg = -float("inf")

    print(f"设备: {agent.device}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print("=" * 55)
    print(f"{'Episode':>8} | {'Reward':>8} | {'Avg(50)':>8} | {'Epsilon':>8}")
    print("=" * 55)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 提前结束给负奖励，加速学习
            shaped_reward = reward if not terminated else -10.0

            agent.store(state, action, shaped_reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        # 定期同步 Target Network
        if episode % agent.target_update_freq == 0:
            agent.sync_target()

        reward_history.append(total_reward)
        avg50 = np.mean(reward_history[-50:])

        if episode % 20 == 0:
            status = " ← best!" if avg50 > best_avg else ""
            print(f"{episode:>8} | {total_reward:>8.1f} | {avg50:>8.1f} | {agent.epsilon:>8.3f}{status}")
            if avg50 > best_avg:
                best_avg = avg50
                torch.save(agent.q_net.state_dict(), "dqn_best.pth")

        # 连续 50 回合平均 ≥ 475 视为解决
        if avg50 >= 475 and episode >= 50:
            print(f"\n✅ 问题解决！Episode {episode}, 最近50回合平均奖励: {avg50:.1f}")
            break

    env.close()
    print(f"\n训练完成。最佳平均奖励: {best_avg:.1f}")
    return agent, reward_history


# ─────────────────────────────────────────
# 5. 评估已训练的模型
# ─────────────────────────────────────────
def evaluate(agent: DQNAgent, num_episodes: int = 10, render: bool = True):
    # 控制一辆小车左右移动，让车上的杆子尽量不倒
    # - Classic Control（经典控制）：`MountainCar-v0`、`Acrobot-v1`、`Pendulum-v1`
    # - Box2D（2D 物理）：`LunarLander-v2`、`BipedalWalker-v3`、`CarRacing-v3`
    # - MuJoCo（连续控制，需额外依赖）：`HalfCheetah-v5`、`Hopper-v5`、`Walker2d-v5`、`Ant-v5`
    # - Atari（像素输入，需额外依赖）：如 `PongNoFrameskip-v4` 等 ALE 系列
    # - Toy Text（离散小环境）：`FrozenLake-v1`、`Taxi-v3`、`CliffWalking-v1`
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    agent.epsilon = 0.0  # 关闭探索，纯贪心

    rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
        print(f"评估 Episode {ep+1}: {total_reward:.0f} 步")

    env.close()
    print(f"\n平均得分: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")


# ─────────────────────────────────────────
# 入口
# ─────────────────────────────────────────
if __name__ == "__main__":
    agent, history = train_dqn(num_episodes=400)
    evaluate(agent, render=True)  # 取消注释可视化评估