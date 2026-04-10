"""
DQN 评估脚本
============
功能:
  - 加载已训练的权重（dqn_best.pth）
  - 在 CartPole-v1 上运行若干回合，统计得分
  - 支持可视化渲染（render=True）

使用:
  python evaluate.py                    # 默认加载 dqn_best.pth，渲染 10 回合
  python evaluate.py --no-render        # 关闭渲染（快速评估）
  python evaluate.py --episodes 20      # 自定义回合数
  python evaluate.py --model my.pth     # 指定权重文件

评估指标:
  - 每回合得分（步数，最大 500）
  - 均值 ± 标准差
"""

import argparse
import numpy as np
import gymnasium as gym

from model import DQNAgent


# ─────────────────────────────────────────────────────────────────
# 评估函数
# ─────────────────────────────────────────────────────────────────
def evaluate(
    model_path : str  = "dqn_best.pth",
    num_episodes: int = 10,
    render     : bool = True,
):
    """
    加载权重并评估 DQNAgent 性能。

    参数:
      model_path   : 权重文件路径
      num_episodes : 评估回合数
      render       : 是否打开可视化窗口

    返回:
      rewards (list[float]) : 每回合得分列表
    """
    env = gym.make(
        "CartPole-v1",
        render_mode="human" if render else None
    )
    state_dim  = env.observation_space.shape[0]          # 4
    action_dim = int(env.action_space.n)                 # type: ignore[attr-defined]  # 2

    # 构建 Agent 并加载权重
    agent = DQNAgent(state_dim, action_dim)
    agent.load(model_path)
    agent.epsilon = 0.0   # 关闭随机探索，纯贪心策略

    print(f"\n{'=' * 45}")
    print(f"  模型路径  : {model_path}")
    print(f"  评估回合  : {num_episodes}")
    print(f"  设备      : {agent.device}")
    print(f"{'=' * 45}")
    print(f"{'Episode':>10} | {'Score':>8} | {'Result':>10}")
    print(f"{'=' * 45}")

    rewards = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

        # 判断单回合是否"完美"（满分 500 步）
        result = "✅ Perfect" if total_reward >= 500 else "❌"
        print(f"{ep:>10} | {total_reward:>8.0f} | {result:>10}")

    env.close()

    # ── 汇总统计 ──────────────────────────────────────────────────
    mean_r = np.mean(rewards)
    std_r  = np.std(rewards)
    max_r  = np.max(rewards)
    min_r  = np.min(rewards)

    print(f"\n{'=' * 45}")
    print(f"  均值  : {mean_r:.1f}")
    print(f"  标准差: {std_r:.1f}")
    print(f"  最高  : {max_r:.0f}")
    print(f"  最低  : {min_r:.0f}")
    print(f"{'=' * 45}")

    if mean_r >= 475:
        print("🏆 模型已达到 CartPole-v1 解决标准（均值 ≥ 475）")
    else:
        print("⚠️  模型尚未达到解决标准，可尝试继续训练。")

    return rewards


# ─────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="评估 DQN CartPole-v1 模型")
    parser.add_argument(
        "--model", type=str, default="dqn_best.pth",
        help="权重文件路径（默认: dqn_best.pth）"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="评估回合数（默认: 10）"
    )
    parser.add_argument(
        "--no-render", dest="render", action="store_false",
        help="关闭可视化渲染"
    )
    parser.set_defaults(render=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path   = args.model,
        num_episodes = args.episodes,
        render       = args.render,
    )

