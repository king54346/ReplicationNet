"""
SAC 训练脚本
============
环境: Pendulum-v1
  状态  : [cos θ, sin θ, θ̇]  (3 维 float)
  动作  : 扭矩 τ ∈ [-2, 2]    (1 维连续)
  奖励  : -(θ² + 0.1*θ̇² + 0.001*τ²)  越接近直立奖励越高（最大 0）
  目标  : 让摆尽量保持直立

Pendulum vs CartPole:
  CartPole (DQN)    离散动作 {0,1}，奖励密集(+1/步)，500步封顶
  Pendulum (SAC)    连续动作 [-2,2]，奖励负数(越直立越接近0)，200步/回合

训练流程:
  1. 前 start_steps 步完全随机探索（填充 Buffer，不训练）
  2. 之后每步交互后做一次梯度更新
  3. 每 log_interval 回合打印一次日志
  4. 自动保存历史最优模型（按最近20回合均值判断）

使用:
  python train.py
"""

import numpy as np
import gymnasium as gym

from model import SACAgent


# ─────────────────────────────────────────────────────────────────
# 训练超参数
# ─────────────────────────────────────────────────────────────────
CONFIG = {
    "num_episodes" : 300,       # 最大训练回合数（每回合 200 步）
    "start_steps"  : 1_000,     # 纯随机探索步数（先填满 Buffer）
    "render"       : False,     # 是否实时渲染
    "lr"           : 3e-4,      # 所有网络学习率
    "gamma"        : 0.99,      # 折扣因子
    "tau"          : 0.005,     # Target 软更新系数
    "alpha_init"   : 0.2,       # 初始温度系数
    "auto_alpha"   : True,      # 自动调节 α
    "buffer_size"  : 100_000,   # ReplayBuffer 容量
    "batch_size"   : 256,       # 训练 batch 大小
    "log_interval" : 10,        # 每隔多少回合打印日志
    "solve_reward" : -200,      # 最近20回合均值高于此视为"解决"
    "save_path"    : "sac_best.pth",
}


# ─────────────────────────────────────────────────────────────────
# 主训练循环
# ─────────────────────────────────────────────────────────────────
def train(cfg: dict | None = None):
    """
    训练 SAC Agent 控制 Pendulum-v1。

    Pendulum-v1 奖励说明:
      reward = -(θ² + 0.1*θ̇² + 0.001*τ²)
        θ   : 摆角（0 = 直立）
        θ̇   : 角速度
        τ   : 施加的扭矩
      每步奖励范围大约 [-16.27, 0]，200步理论最优约为 -100~0

    SAC 与 DQN 训练差异:
      DQN    : 每步存 → 每步训练（Buffer 够后）；ε 显式衰减控制探索
      SAC    : 前 start_steps 步随机探索；之后每步存+训练；α 自动调控探索
    """
    if cfg is None:
        cfg = CONFIG

    env = gym.make(
        "Pendulum-v1",
        render_mode="human" if cfg["render"] else None
    )

    state_dim   = env.observation_space.shape[0]    # 3
    action_dim  = env.action_space.shape[0]          # 1
    action_scale = float(env.action_space.high[0])  # type: ignore[attr-defined]  # 2.0

    agent = SACAgent(
        state_dim    = state_dim,
        action_dim   = action_dim,
        action_scale = action_scale,
        lr           = cfg["lr"],
        gamma        = cfg["gamma"],
        tau          = cfg["tau"],
        alpha_init   = cfg["alpha_init"],
        auto_alpha   = cfg["auto_alpha"],
        buffer_size  = cfg["buffer_size"],
        batch_size   = cfg["batch_size"],
    )

    reward_history = []
    best_avg       = -float("inf")
    total_steps    = 0       # 全局步数计数（用于判断是否结束随机探索）

    print(f"设备       : {agent.device}")
    print(f"状态维度   : {state_dim}  |  动作维度: {action_dim}")
    print(f"动作范围   : [{-action_scale}, {action_scale}]")
    print(f"随机探索步 : {cfg['start_steps']}")
    print("=" * 65)
    print(f"{'Episode':>8} | {'Reward':>8} | {'Avg(20)':>8} | {'Alpha':>7} | {'Steps':>7}")
    print("=" * 65)

    for episode in range(1, cfg["num_episodes"] + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        last_info = {}

        while not done:
            total_steps += 1

            # 前 start_steps 步：完全随机动作（充分探索状态空间）
            # 之后：用当前策略采样（随机策略，保留探索性）
            if total_steps < cfg["start_steps"]:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Pendulum 不会真正 terminated（只会 truncated=True 在 200 步）
            # 所以 done_flag 只在真正失败时为 1（这里 Pendulum 始终为 truncated）
            done_flag = float(terminated)

            agent.store(state, action, reward, next_state, done_flag)

            # 开始训练（Buffer 中有足够样本后）
            info = agent.train()
            if info:
                last_info = info

            state         = next_state
            total_reward += reward

        reward_history.append(total_reward)
        avg20 = np.mean(reward_history[-20:])

        # ── 日志 ──────────────────────────────────────────────────
        if episode % cfg["log_interval"] == 0:
            alpha_val = last_info.get("alpha", cfg["alpha_init"])
            tag = " ← best!" if avg20 > best_avg else ""
            print(
                f"{episode:>8} | {total_reward:>8.1f} | "
                f"{avg20:>8.1f} | {alpha_val:>7.4f} | {total_steps:>7}{tag}"
            )
            if avg20 > best_avg:
                best_avg = avg20
                agent.save(cfg["save_path"])

        # ── 提前结束 ──────────────────────────────────────────────
        # Pendulum-v1 没有官方"解决"标准，这里用 -200 作为参考
        if avg20 >= cfg["solve_reward"] and episode >= 20:
            print(
                f"\n✅ 达到目标！Episode {episode}，"
                f"最近20回合平均奖励: {avg20:.1f}"
            )
            break

    env.close()
    print(f"\n训练完成。最佳 20 回合平均奖励: {best_avg:.1f}")
    return agent, reward_history


# ─────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent, history = train()

