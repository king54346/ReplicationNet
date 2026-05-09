"""
train.py — Double Dueling DQN 训练脚本（驱动真实 Chrome Dino）

用法：
    python train.py                              # 全新训练
    python train.py --resume                     # 自动续训（加载 latest_ckpt.pth）
    python train.py --resume --ckpt checkpoints/model_ep200.pth
    python train.py --episodes 1000 --headless
    python train.py --delay 0.04 --episodes 500
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401 (used in optimize)

from model import ChromeDinoEnv, DQN, ReplayBuffer

# ──────────────────────────────────────────────
# 超参数
# ──────────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR              = 5e-4
GAMMA           = 0.99
EPSILON_START   = 1.0
EPSILON_MIN     = 0.05
EPSILON_DECAY   = 0.992
BATCH_SIZE      = 64
REPLAY_CAPACITY = 50_000
REPLAY_START    = 500
TARGET_UPDATE   = 300
GRAD_CLIP       = 10.0
CHECKPOINT_DIR  = "checkpoints"
SAVE_INTERVAL   = 100

# 断点续训的元数据文件（保存 epsilon、episode、total_steps、best_score）
META_FILE       = os.path.join(CHECKPOINT_DIR, "train_meta.json")
LATEST_CKPT     = os.path.join(CHECKPOINT_DIR, "latest_ckpt.pth")


# ──────────────────────────────────────────────
# 保存 / 加载断点
# ──────────────────────────────────────────────
def save_checkpoint(path: str,
                    policy: DQN,
                    target: DQN,
                    opt: torch.optim.Optimizer,
                    epsilon: float,
                    episode: int,
                    total_steps: int,
                    best_score: int,
                    history: list[int]):
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


def load_checkpoint(path: str,
                    policy: DQN,
                    target: DQN,
                    opt: torch.optim.Optimizer):
    ckpt = torch.load(path, map_location=DEVICE)

    # 兼容旧格式：文件直接是 state_dict（无 "policy" 键）
    if "policy" not in ckpt:
        print("  [兼容模式] 检测到旧格式权重，仅加载网络参数，其余从默认值开始。")
        policy.load_state_dict(ckpt)
        target.load_state_dict(ckpt)
        return EPSILON_MIN, 0, 0, 0, []

    policy.load_state_dict(ckpt["policy"])
    target.load_state_dict(ckpt["target"])
    opt.load_state_dict(ckpt["optimizer"])
    return (
        ckpt["epsilon"],
        ckpt["episode"],
        ckpt["total_steps"],
        ckpt["best_score"],
        ckpt.get("history", []),
    )


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

    policy = DQN(ChromeDinoEnv.STATE_DIM, ChromeDinoEnv.ACTION_DIM).to(DEVICE)
    target = DQN(ChromeDinoEnv.STATE_DIM, ChromeDinoEnv.ACTION_DIM).to(DEVICE)
    target.eval()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    buf = ReplayBuffer(REPLAY_CAPACITY)

    # ── 断点续训 ──────────────────────────────
    epsilon     = EPSILON_START
    start_ep    = 1
    ep          = 1          # 保证 finally 块中 ep 始终有值
    total_steps = 0
    best_score  = 0
    history: list[int] = []

    if resume:
        path = ckpt_path if ckpt_path else LATEST_CKPT
        if os.path.exists(path):
            epsilon, start_ep, total_steps, best_score, history = \
                load_checkpoint(path, policy, target, opt)
            start_ep += 1          # 从下一个 episode 继续
            print(f"[续训] 已加载: {path}")
            print(f"       从 Episode {start_ep} 继续 | "
                  f"ε={epsilon:.4f} | best={best_score} | "
                  f"已训步数={total_steps}")
        else:
            print(f"[警告] 找不到断点文件 {path}，改为全新训练。")
    else:
        target.load_state_dict(policy.state_dict())

    end_ep = start_ep + max_episodes - 1
    print(f"\n设备: {DEVICE}  |  训练 Episode {start_ep}→{end_ep}  |  "
          f"headless: {headless}  |  step_delay: {step_delay}s")

    env = ChromeDinoEnv(headless=headless, step_delay=step_delay)

    try:
        for ep in range(start_ep, end_ep + 1):
            state         = env.reset()
            ep_reward     = 0.0
            action_counts = [0, 0, 0]

            while True:
                if np.random.random() < epsilon:
                    action = np.random.randint(ChromeDinoEnv.ACTION_DIM)
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

            # 保存最优权重（仅网络参数，供 eval 使用）
            if score > best_score:
                best_score = score
                torch.save(policy.state_dict(),
                           os.path.join(CHECKPOINT_DIR, "best_model.pth"))

            # 打印日志
            if ep % 5 == 0:
                avg20      = np.mean(history[-20:])
                total_acts = max(sum(action_counts), 1)
                noop, jump, duck = action_counts
                print(
                    f"Ep {ep:5d} | Score {score:5d} | Best {best_score:5d} | "
                    f"Avg20 {avg20:6.0f} | ε {epsilon:.4f} | "
                    f"Noop {noop/total_acts:.0%} "
                    f"Jump {jump/total_acts:.0%} "
                    f"Duck {duck/total_acts:.0%}"
                )

            # 定期保存完整断点
            if ep % SAVE_INTERVAL == 0:
                ep_ckpt = os.path.join(CHECKPOINT_DIR, f"model_ep{ep}.pth")
                save_checkpoint(ep_ckpt, policy, target, opt,
                                epsilon, ep, total_steps, best_score, history)
                # 同时覆盖 latest_ckpt 方便 --resume 快速找到
                save_checkpoint(LATEST_CKPT, policy, target, opt,
                                epsilon, ep, total_steps, best_score, history)
                print(f"  [保存断点] {ep_ckpt}")

    except KeyboardInterrupt:
        print("\n手动中断，保存断点...")
    finally:
        # 退出时始终保存 latest_ckpt，下次 --resume 可接续
        save_checkpoint(LATEST_CKPT, policy, target, opt,
                        epsilon, ep if 'ep' in dir() else start_ep,
                        total_steps, best_score, history)
        torch.save(policy.state_dict(),
                   os.path.join(CHECKPOINT_DIR, "final_model.pth"))
        print(f"[已保存断点] {LATEST_CKPT}")
        env.close()

    print(f"\n训练结束 | 最高分: {best_score} | 总步数: {total_steps}")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chrome Dino DQN 训练")
    parser.add_argument("--episodes", type=int, default=500,
                        help="本次训练的 episode 数（续训时叠加，默认 500）")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true",
                        help="从上次断点继续训练")
    parser.add_argument("--ckpt", type=str, default="",
                        help="指定续训的断点文件（默认自动找 latest_ckpt.pth）")
    args = parser.parse_args()

    train(args.episodes, args.headless, args.delay, args.resume, args.ckpt)