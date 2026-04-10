"""
SAC 评估脚本
============
功能:
  - 加载已训练的权重（sac_best.pth）
  - 在 Pendulum-v1 上运行若干回合，统计得分
  - 支持可视化渲染（--render）

Pendulum-v1 奖励参考:
  随机策略      : 约 -1200 ~ -1500 / 回合
  训练中的策略  : 约 -400  ~ -800  / 回合
  较好的策略    : 约 -150  ~ -300  / 回合
  接近最优策略  : 约 -100  ~ -200  / 回合（每步奖励接近 0）

使用:
  python evaluate.py                        # 渲染 5 回合
  python evaluate.py --no-render --ep 20    # 不渲染评估 20 回合
  python evaluate.py --model sac_best.pth   # 指定权重文件
"""

import argparse
import numpy as np
import gymnasium as gym

from model import SACAgent


# ─────────────────────────────────────────────────────────────────
# 评估函数
# ─────────────────────────────────────────────────────────────────
def evaluate(
    model_path   : str  = "sac_best.pth",
    num_episodes : int  = 5,
    render       : bool = True,
):
    """
    加载权重并评估 SAC Agent 性能。

    评估策略：deterministic=True（使用均值，关闭随机采样）
              → 去除随机性，得到最稳定的表现

    参数:
      model_path   : 权重文件路径
      num_episodes : 评估回合数
      render       : 是否打开可视化窗口

    返回:
      rewards (list[float]) : 每回合累计奖励
    """
    env = gym.make(
        "Pendulum-v1",
        render_mode="human" if render else None
    )

    state_dim    = env.observation_space.shape[0]       # 3
    action_dim   = env.action_space.shape[0]             # 1
    action_scale = float(env.action_space.high[0])  # type: ignore[attr-defined]  # 2.0

    agent = SACAgent(
        state_dim    = state_dim,
        action_dim   = action_dim,
        action_scale = action_scale,
    )
    agent.load(model_path)

    print(f"\n{'=' * 50}")
    print(f"  模型路径  : {model_path}")
    print(f"  评估回合  : {num_episodes}")
    print(f"  动作范围  : [{-action_scale:.1f}, {action_scale:.1f}]")
    print(f"  设备      : {agent.device}")
    print(f"{'=' * 50}")
    print(f"{'Episode':>10} | {'Score':>10} | {'Avg/Step':>10}")
    print(f"{'=' * 50}")

    rewards = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            # 评估用确定性动作（均值），去掉随机性
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        avg_per_step = total_reward / steps if steps > 0 else 0.0
        print(f"{ep:>10} | {total_reward:>10.1f} | {avg_per_step:>10.3f}")

    env.close()

    # ── 汇总统计 ──────────────────────────────────────────────────
    mean_r = np.mean(rewards)
    std_r  = np.std(rewards)
    max_r  = np.max(rewards)
    min_r  = np.min(rewards)

    print(f"\n{'=' * 50}")
    print(f"  均值     : {mean_r:.1f}")
    print(f"  标准差   : {std_r:.1f}")
    print(f"  最高     : {max_r:.1f}")
    print(f"  最低     : {min_r:.1f}")
    print(f"{'=' * 50}")

    # Pendulum-v1 参考评级
    if mean_r >= -200:
        print("🏆 优秀！策略已接近最优（均值 ≥ -200）")
    elif mean_r >= -400:
        print("✅ 良好！策略已基本收敛（均值 ≥ -400）")
    elif mean_r >= -700:
        print("⚠️  一般，策略仍在改善中（均值 ≥ -700）")
    else:
        print("❌ 较差，建议继续训练（均值 < -700）")

    return rewards


# ─────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="评估 SAC Pendulum-v1 模型")
    parser.add_argument(
        "--model", type=str, default="sac_best.pth",
        help="权重文件路径（默认: sac_best.pth）"
    )
    parser.add_argument(
        "--ep", type=int, default=5,
        help="评估回合数（默认: 5）"
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
        num_episodes = args.ep,
        render       = args.render,
    )

