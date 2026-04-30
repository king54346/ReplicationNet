"""
评估已训练的 Dueling DQN 模型

用法：
  python eval.py --model_path ./checkpoints/dueling_dqn_final.pt
  python eval.py --model_path ./checkpoints/dueling_dqn_final.pt --episodes 50 --no_render
"""

import argparse
import numpy as np
import torch
import gymnasium as gym

from model import DuelingDQN


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="./checkpoints/dueling_dqn_final.pt")
    p.add_argument("--episodes",   type=int, default=10)
    p.add_argument("--hidden",     type=int, default=1024)
    p.add_argument("--no_render",  action="store_true", help="不显示画面")
    return p.parse_args()


def evaluate():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    render_mode = None if args.no_render else "human"
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = DuelingDQN(obs_dim, act_dim, args.hidden).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"已加载模型: {args.model_path}\n")

    all_rewards = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = model(obs_t).argmax(dim=-1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        all_rewards.append(total_reward)
        status = "成功" if total_reward >= 200 else ("坠毁" if total_reward < 0 else "一般")
        print(f"  回合 {ep:>3d} | 得分 {total_reward:>8.1f} | 步数 {steps:>4d} | {status}")

    env.close()

    print("\n── 评估统计 ──────────────────────────────")
    print(f"  回合数  : {args.episodes}")
    print(f"  均分    : {np.mean(all_rewards):.1f}")
    print(f"  最高分  : {np.max(all_rewards):.1f}")
    print(f"  最低分  : {np.min(all_rewards):.1f}")
    print(f"  标准差  : {np.std(all_rewards):.1f}")
    success = sum(r >= 200 for r in all_rewards)
    print(f"  成功率  : {success}/{args.episodes} ({100*success/args.episodes:.0f}%)")


if __name__ == "__main__":
    evaluate()
