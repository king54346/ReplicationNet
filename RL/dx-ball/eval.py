"""
eval.py — DX-Ball DQN 评估 / 可视化

用法：
    python eval.py                               # 加载 best_model，跑 5 局
    python eval.py --model checkpoints/model_ep200.pth --episodes 10
    python eval.py --headless --episodes 50      # 无界面批量评估
    python eval.py --plot                        # 画训练曲线
    python eval.py --delay 0.03                  # 加速回放
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from model import DQN, DXBallEnv, STACK_N

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"


# ──────────────────────────────────────────────
# 加载模型（兼容新旧两种格式）
# ──────────────────────────────────────────────
def load_model(model_path: str) -> DQN:
    if not os.path.exists(model_path):
        print(f"[错误] 找不到模型: {model_path}")
        print("请先运行: python train.py")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location=DEVICE)
    state_dict = ckpt["policy"] if "policy" in ckpt else ckpt

    model = DQN(STACK_N, DXBallEnv.ACTION_DIM).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ──────────────────────────────────────────────
# 单局评估
# ──────────────────────────────────────────────
def run_episode(model: DQN, env: DXBallEnv):
    state        = env.reset()
    total_reward = 0.0
    steps        = 0
    action_log   = [0, 0, 0]

    while True:
        with torch.no_grad():
            sv     = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
            action = model(sv).argmax().item()

        state, reward, done, info = env.step(action)
        total_reward += reward
        steps        += 1
        action_log[action] += 1

        if done:
            break

    return env.score, total_reward, steps, action_log


# ──────────────────────────────────────────────
# 批量评估
# ──────────────────────────────────────────────
def evaluate(model_path: str, num_episodes: int,
             headless: bool, delay: float):
    model = load_model(model_path)
    print(f"已加载: {model_path}  (设备: {DEVICE})")

    env    = DXBallEnv(headless=headless, step_delay=delay)
    scores = []

    print(f"\n{'Ep':>4}  {'Score':>7}  {'Reward':>9}  {'Steps':>6}  "
          f"{'Noop':>6}  {'Left':>6}  {'Right':>6}")
    print("-" * 58)

    try:
        for ep in range(1, num_episodes + 1):
            score, reward, steps, acts = run_episode(model, env)
            scores.append(score)
            total_a = max(sum(acts), 1)
            print(
                f"{ep:>4}  {score:>7}  {reward:>9.1f}  {steps:>6}  "
                f"{acts[0]/total_a:>5.0%}  "
                f"{acts[1]/total_a:>5.0%}  "
                f"{acts[2]/total_a:>5.0%}"
            )
    except KeyboardInterrupt:
        print("\n中断评估。")
    finally:
        env.close()

    if scores:
        print("\n" + "=" * 45)
        print(f"共评估:  {len(scores)} 局")
        print(f"平均分:  {np.mean(scores):.1f}")
        print(f"最高分:  {max(scores)}")
        print(f"最低分:  {min(scores)}")
        print(f"标准差:  {np.std(scores):.1f}")
        print("=" * 45)

    return scores


# ──────────────────────────────────────────────
# 绘制训练曲线
# ──────────────────────────────────────────────
def plot_curve(history_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[错误] pip install matplotlib")
        return

    if not os.path.exists(history_path):
        print(f"[错误] 找不到: {history_path}")
        return

    with open(history_path) as f:
        scores = json.load(f)

    ep  = np.arange(1, len(scores) + 1)
    win = min(50, max(len(scores) // 10, 1))
    ma  = np.convolve(scores, np.ones(win) / win, mode="valid")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(ep, scores, alpha=0.25, color="#4C9BE8", label="每局分数")
    axes[0].plot(ep[win-1:], ma, color="#E84C4C", lw=2,
                 label=f"{win}-ep 移动均值")
    axes[0].set_ylabel("Score")
    axes[0].set_title("DX-Ball Double Dueling DQN 训练曲线")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    best_so_far = np.maximum.accumulate(scores)
    axes[1].plot(ep, best_so_far, color="#4CAF50", lw=2, label="历史最高分")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Best Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(CHECKPOINT_DIR, "training_curve.png")
    plt.savefig(out, dpi=150)
    print(f"已保存: {out}")
    plt.show()


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DX-Ball DQN 评估")
    parser.add_argument("--model",
                        default=os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--plot", action="store_true",
                        help="画训练曲线（需要 history.json）")
    args = parser.parse_args()

    if args.plot:
        plot_curve(os.path.join(CHECKPOINT_DIR, "history.json"))
    else:
        evaluate(args.model, args.episodes, args.headless, args.delay)
