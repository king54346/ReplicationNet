"""
train.py — DX-Ball Double Dueling DQN 训练脚本

用法：
    python train.py                              # 全新训练
    python train.py --resume                     # 自动续训 (latest_ckpt.pth)
    python train.py --resume --ckpt checkpoints/model_ep200.pth
    python train.py --headless --episodes 1000
    python train.py --delay 0.04
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from model import DQN, DXBallEnv, ReplayBuffer, STACK_N

# ──────────────────────────────────────────────
# 超参数
# ──────────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR              = 1e-4           # CNN 用更小学习率
GAMMA           = 0.99
EPSILON_START   = 1.0
EPSILON_MIN     = 0.05
EPSILON_DECAY   = 0.990
BATCH_SIZE      = 32
REPLAY_CAPACITY = 30_000
REPLAY_START    = 1_000
TARGET_UPDATE   = 500
GRAD_CLIP       = 10.0
CHECKPOINT_DIR  = "checkpoints"
SAVE_INTERVAL   = 50
LATEST_CKPT     = os.path.join(CHECKPOINT_DIR, "latest_ckpt.pth")


# ──────────────────────────────────────────────
# 断点保存 / 加载
# ──────────────────────────────────────────────
def save_checkpoint(path, policy, target, opt,
                    epsilon, episode, total_steps, best_score, history):
    torch.save({
        "policy":      policy.state_dict(),
        "target":      target.state_dict(),
        "optimizer":   opt.state_dict(),
        "epsilon":     epsilon,
        "episode":     episode,
        "total_steps": total_steps,
        "best_score":  best_score,
        "history":     history,
    }, path)


def load_checkpoint(path, policy, target, opt):
    ckpt = torch.load(path, map_location=DEVICE)
    # 兼容旧格式（裸 state_dict）
    if "policy" not in ckpt:
        print("  [兼容模式] 旧格式权重，仅加载网络参数。")
        policy.load_state_dict(ckpt)
        target.load_state_dict(ckpt)
        return EPSILON_MIN, 0, 0, 0, []
    policy.load_state_dict(ckpt["policy"])
    target.load_state_dict(ckpt["target"])
    opt.load_state_dict(ckpt["optimizer"])
    return (ckpt["epsilon"], ckpt["episode"],
            ckpt["total_steps"], ckpt["best_score"],
            ckpt.get("history", []))


# ──────────────────────────────────────────────
# Double DQN 优化步
# ──────────────────────────────────────────────
def optimize(policy: DQN, target: DQN,
             opt: torch.optim.Optimizer,
             buf: ReplayBuffer) -> float:
    s, a, r, ns, d = buf.sample(BATCH_SIZE)

    s  = torch.from_numpy(s).to(DEVICE)
    a  = torch.from_numpy(a).to(DEVICE)
    r  = torch.from_numpy(r).to(DEVICE)
    ns = torch.from_numpy(ns).to(DEVICE)
    d  = torch.from_numpy(d).to(DEVICE)

    q_cur = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        best_a = policy(ns).argmax(dim=1, keepdim=True)
        q_next = target(ns).gather(1, best_a).squeeze(1)
        q_tgt  = r + GAMMA * q_next * (1.0 - d)

    loss = F.smooth_l1_loss(q_cur, q_tgt)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
    opt.step()
    return loss.item()


# ──────────────────────────────────────────────
# 主训练循环
# ──────────────────────────────────────────────
def train(max_episodes: int, headless: bool, step_delay: float,
          resume: bool, ckpt_path: str):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    action_dim = DXBallEnv.ACTION_DIM
    policy = DQN(STACK_N, action_dim).to(DEVICE)
    target = DQN(STACK_N, action_dim).to(DEVICE)
    target.eval()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    buf = ReplayBuffer(REPLAY_CAPACITY)

    epsilon     = EPSILON_START
    ep          = 1
    start_ep    = 1
    total_steps = 0
    best_score  = 0
    history: list[int] = []

    if resume:
        path = ckpt_path or LATEST_CKPT
        if os.path.exists(path):
            epsilon, start_ep, total_steps, best_score, history = \
                load_checkpoint(path, policy, target, opt)
            start_ep += 1
            print(f"[续训] {path}  Episode {start_ep}  ε={epsilon:.4f}  best={best_score}")
        else:
            print(f"[警告] 找不到 {path}，全新训练。")
            target.load_state_dict(policy.state_dict())
    else:
        target.load_state_dict(policy.state_dict())

    end_ep = start_ep + max_episodes - 1
    print(f"设备: {DEVICE}  Episode {start_ep}→{end_ep}  headless={headless}")

    env = DXBallEnv(headless=False, step_delay=step_delay)  # pyautogui 不支持 headless

    try:
        for ep in range(start_ep, end_ep + 1):
            state         = env.reset()
            ep_reward     = 0.0
            action_counts = [0, 0, 0]

            while True:
                if np.random.random() < epsilon:
                    action = np.random.randint(action_dim)
                else:
                    with torch.no_grad():
                        sv     = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
                        action = policy(sv).argmax().item()

                next_state, reward, done, info = env.step(action)
                buf.push(state, action, reward, next_state, done)
                state          = next_state
                ep_reward     += reward
                total_steps   += 1
                action_counts[action] += 1

                if len(buf) >= REPLAY_START:
                    optimize(policy, target, opt, buf)

                if total_steps % TARGET_UPDATE == 0:
                    target.load_state_dict(policy.state_dict())

                if done:
                    break

            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            score   = env.score
            history.append(score)

            if score > best_score:
                best_score = score
                torch.save(policy.state_dict(),
                           os.path.join(CHECKPOINT_DIR, "best_model.pth"))

            if ep % 5 == 0:
                avg20      = np.mean(history[-20:])
                total_acts = max(sum(action_counts), 1)
                n, l, r_   = action_counts
                print(
                    f"Ep {ep:5d} | Score {score:6d} | Best {best_score:6d} | "
                    f"Avg20 {avg20:7.0f} | ε {epsilon:.4f} | "
                    f"Noop {n/total_acts:.0%} "
                    f"Left {l/total_acts:.0%} "
                    f"Right {r_/total_acts:.0%} | "
                    f"Buf {len(buf)}"
                )

            if ep % SAVE_INTERVAL == 0:
                ep_ckpt = os.path.join(CHECKPOINT_DIR, f"model_ep{ep}.pth")
                for path in (ep_ckpt, LATEST_CKPT):
                    save_checkpoint(path, policy, target, opt,
                                    epsilon, ep, total_steps, best_score, history)
                print(f"  [保存] {ep_ckpt}")

    except KeyboardInterrupt:
        print("\n手动中断，保存断点...")
    finally:
        save_checkpoint(LATEST_CKPT, policy, target, opt,
                        epsilon, ep, total_steps, best_score, history)
        torch.save(policy.state_dict(),
                   os.path.join(CHECKPOINT_DIR, "final_model.pth"))
        print(f"[断点已保存] {LATEST_CKPT}")
        env.close()

    print(f"\n训练结束 | 最高分: {best_score} | 总步数: {total_steps}")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DX-Ball DQN 训练")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--headless", action="store_true",
                        help="（pyautogui 模式下此参数无效，已忽略）")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="每步间隔（秒），默认 0.05")
    parser.add_argument("--resume", action="store_true",
                        help="从断点继续训练")
    parser.add_argument("--ckpt", type=str, default="",
                        help="指定断点文件（默认 latest_ckpt.pth）")
    args = parser.parse_args()

    train(args.episodes, args.headless, args.delay, args.resume, args.ckpt)
