"""
DQN 训练脚本
============
功能:
  - 在 CartPole-v1 环境中训练 DQNAgent
  - 每 20 回合打印进度 (回合奖励 / 最近50回合均值 / ε)
  - 自动保存历史最优模型权重到 dqn_best.pth
  - 连续 50 回合平均奖励 ≥ 475 时提前结束（视为解决）

使用:
  python train.py

训练技巧:
  1. Experience Replay  : 随机采样打破时序相关性
  2. Target Network     : 固定 TD 目标，避免追着移动靶子发散
  3. Reward Shaping     : 提前终止（杆倒下）时给 -10 惩罚，加速学习信号传播
  4. ε-greedy 衰减      : 从纯随机探索逐渐过渡到贪心利用
  5. 梯度裁剪           : clip_grad_norm 防止梯度爆炸
"""

import numpy as np
import gymnasium as gym

from model import DQNAgent


# ─────────────────────────────────────────────────────────────────
# 训练超参数
# ─────────────────────────────────────────────────────────────────
CONFIG = dict(
    num_episodes       = 400,    # 最大训练回合数
    render             = False,  # 是否实时渲染（会拖慢速度）
    lr                 = 1e-3,   # Adam 学习率
    gamma              = 0.99,   # 折扣因子
    epsilon_start      = 1.0,    # 初始探索率
    epsilon_end        = 0.05,   # 最小探索率
    epsilon_decay      = 0.995,  # 每回合衰减系数
    buffer_size        = 10_000, # ReplayBuffer 容量
    batch_size         = 64,     # 每次训练采样量
    target_update_freq = 10,     # Target Network 同步间隔（回合）
    solve_threshold    = 475,    # 认为"解决"所需的最近50回合均值
    save_path          ="dqn_best.pth",
)


# ─────────────────────────────────────────────────────────────────
# 主训练循环
# ─────────────────────────────────────────────────────────────────
def train(cfg: dict | None = None):
    """
    训练 DQNAgent。

    CartPole-v1 环境说明:
      状态  : [车位置, 车速度, 杆角度, 杆角速度]  (4 维 float)
      动作  : {0: 向左, 1: 向右}
      奖励  : 每存活一步 +1，最多 500 步/回合
      终止  : 杆倾斜 > 12° 或 车位移 > 2.4

    Reward Shaping:
      正常存活步   → shaped_reward = +1.0
      杆倒（终止） → shaped_reward = -10.0（加速向前传播惩罚信号）
    """
    if cfg is None:
        cfg = CONFIG
    env = gym.make(
        "CartPole-v1",
        render_mode="human" if cfg["render"] else None
    )
    state_dim  = env.observation_space.shape[0]          # 4
    action_dim = int(env.action_space.n)                 # type: ignore[attr-defined]  # 2

    agent = DQNAgent(
        state_dim          = state_dim,
        action_dim         = action_dim,
        lr                 = cfg["lr"],
        gamma              = cfg["gamma"],
        epsilon_start      = cfg["epsilon_start"],
        epsilon_end        = cfg["epsilon_end"],
        epsilon_decay      = cfg["epsilon_decay"],
        buffer_size        = cfg["buffer_size"],
        batch_size         = cfg["batch_size"],
        target_update_freq = cfg["target_update_freq"],
    )

    reward_history = []
    best_avg       = -float("inf")

    print(f"设备: {agent.device}")
    print(f"状态维度: {state_dim}  |  动作维度: {action_dim}")
    print("=" * 60)
    print(f"{'Episode':>8} | {'Reward':>8} | {'Avg(50)':>8} | {'Epsilon':>8}")
    print("=" * 60)

    for episode in range(1, cfg["num_episodes"] + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        # ── 单回合交互 ──────────────────────────────────────────────
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Reward Shaping: 杆倒下时给负惩罚
            shaped_reward = reward if not terminated else -10.0

            agent.store(state, action, shaped_reward, next_state, done)
            agent.train()      # Buffer 不足时自动跳过

            state         = next_state
            total_reward += reward  # 累计原始奖励（用于评估）

        # ── 回合结束后更新 ──────────────────────────────────────────
        agent.decay_epsilon()

        # 每 target_update_freq 回合同步一次 Target Network
        if episode % agent.target_update_freq == 0:
            agent.sync_target()

        reward_history.append(total_reward)
        avg50 = np.mean(reward_history[-50:])

        # ── 日志 & 保存最优模型 ─────────────────────────────────────
        if episode % 20 == 0:
            tag = " ← best!" if avg50 > best_avg else ""
            print(
                f"{episode:>8} | {total_reward:>8.1f} | "
                f"{avg50:>8.1f} | {agent.epsilon:>8.3f}{tag}"
            )
            if avg50 > best_avg:
                best_avg = avg50
                agent.save(cfg["save_path"])

        # ── 提前结束条件 ────────────────────────────────────────────
        if avg50 >= cfg["solve_threshold"] and episode >= 50:
            print(
                f"\n✅ 问题已解决！"
                f"Episode {episode}，最近50回合平均奖励: {avg50:.1f}"
            )
            break

    env.close()
    print(f"\n训练完成。最佳 50 回合平均奖励: {best_avg:.1f}")
    return agent, reward_history


# ─────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent, history = train()

