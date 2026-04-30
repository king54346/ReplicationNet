"""
Dueling DQN 训练 LunarLander-v3

用法：
  python train.py
  python train.py --total_steps 500000 --hidden 256
"""

import os
import argparse
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from model import DuelingDQN


# ──────────────────────────────────────────────────────────
# 参数
# ──────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total_steps",    type=int,   default=500_000)
    p.add_argument("--hidden",         type=int,   default=256)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--gamma",          type=float, default=0.99)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--buffer_size",    type=int,   default=50_000)
    p.add_argument("--learn_start",    type=int,   default=1_000,
                   help="开始学习前先收集多少步")
    p.add_argument("--learn_freq",     type=int,   default=4,
                   help="每隔多少步更新一次网络")
    p.add_argument("--target_update",  type=int,   default=1_000,
                   help="每隔多少步同步目标网络")
    p.add_argument("--eps_start",      type=float, default=1.0)
    p.add_argument("--eps_end",        type=float, default=0.01)
    p.add_argument("--eps_decay",      type=int,   default=200_000,
                   help="epsilon 线性衰减的步数")
    p.add_argument("--save_dir",       type=str,   default="./checkpoints")
    p.add_argument("--save_interval",  type=int,   default=100_000)
    p.add_argument("--log_interval",   type=int,   default=20,
                   help="每隔多少回合打印一次")
    return p.parse_args()


# ──────────────────────────────────────────────────────────
# 经验回放缓冲区
# ──────────────────────────────────────────────────────────
Transition = collections.namedtuple(
    "Transition", ["obs", "action", "reward", "next_obs", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(action,   dtype=np.int64),
            np.array(reward,   dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(done,     dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ──────────────────────────────────────────────────────────
# 主训练循环
# ──────────────────────────────────────────────────────────
def train():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    env = gym.make("LunarLander-v3")
    # LunarLander-v3 的 observation 是 8 维连续向量，action 是 4 个离散动作
    # 主引擎  (2) 会产生向上的推力，左右引擎 (1/3) 会产生水平推力，0 是不点火
    obs_dim = env.observation_space.shape[0]   # 8
    act_dim = env.action_space.n               # 4

    # ── 在线网络 + 目标网络 ──
    online_net = DuelingDQN(obs_dim, act_dim, args.hidden).to(device)
    target_net = DuelingDQN(obs_dim, act_dim, args.hidden).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=args.lr)
    buffer    = ReplayBuffer(args.buffer_size)

    # ── 辅助函数 ──
    # ε 从 eps_start 线性衰减到 eps_end，衰减过程持续 eps_decay 步，之后保持 eps_end 不变。
    def epsilon(step: int) -> float:
        ratio = min(step / args.eps_decay, 1.0)
        return args.eps_start + ratio * (args.eps_end - args.eps_start)

    # 选择动作：以 ε 的概率随机选动作，否则选 Q 值最大的动作
    def select_action(obs_np: np.ndarray, step: int) -> int:
        if random.random() < epsilon(step):
            return env.action_space.sample()
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return online_net(obs_t).argmax(dim=-1).item()
    # 关键代码
    # 只更新 online_net，target_net 完全不动
    # 1. optimizer 创建时绑定的是 online_net 2. loss 是通过 online_net 的输出算出来的
    def update():
        obs, action, reward, next_obs, done = buffer.sample(args.batch_size)
        obs_t      = torch.tensor(obs,      device=device)
        action_t   = torch.tensor(action,   device=device)
        reward_t   = torch.tensor(reward,   device=device)
        next_obs_t = torch.tensor(next_obs, device=device)
        done_t     = torch.tensor(done,     device=device)
        # online_net(obs_t) 输出当前状态所有动作的 Q 值
        # 取出当时实际执行的动作的 Q 值
        q_values = online_net(obs_t).gather(1, action_t.unsqueeze(1)).squeeze(1)

        # Double DQN：用在线网络选动作，用目标网络估价值
        with torch.no_grad():
            # 这个块里的所有计算不建立计算图，不记录梯度
            # 选出下一步的Q值最大的动作
            next_actions = online_net(next_obs_t).argmax(dim=-1, keepdim=True)
            # 目标网络估计这些动作的Q值
            next_q       = target_net(next_obs_t).gather(1, next_actions).squeeze(1)
            # 计算 TD Target = reward + gamma * Q(next_obs) * (1 - float(terminated))
            # TD Target = 这步真实奖励 + 网络对未来的估计
            target_q     = reward_t + args.gamma * next_q * (1 - done_t)
        # loss 用target net 和 online net 之间的Q 值差距来衡量
        loss = nn.functional.smooth_l1_loss(q_values, target_q)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
        optimizer.step()
        return loss.item()

    # ── 训练 ──
    obs, _ = env.reset()
    ep_reward   = 0.0
    ep_count    = 0
    recent_rew  = collections.deque(maxlen=20)
    losses      = []
    last_save   = 0

    print(f"开始训练，目标步数: {args.total_steps:,}")

    for step in range(1, args.total_steps + 1):
        action = select_action(obs, step)
        # LunarLander-v3 的 step() 返回 (next_obs, reward, terminated, truncated, info)
        # terminated 是因为成功着陆或坠毁导致的 episode 结束，truncated 是因为达到最大步数限制导致的 episode 结束。
        # 我们把两者都视为 episode 结束的条件。
        # next_obs 是一个 8 维连续向量，reward 是一个标量，done 是一个布尔值。
        # next_obs 环境返回新的观测值
        # next_obs = [
        #     0.02,   # state[0] x 位置（0=中心，正=右，负=左）
        #    -0.31,   # state[1] y 位置（0=地面，正=上方）
        #     0.05,   # state[2] x 速度
        #    -0.20,   # state[3] y 速度（负=向下）
        #     0.01,   # state[4] 角度（0=竖直，正=右倾）
        #    -0.03,   # state[5] 角速度
        #     0.0,    # state[6] 左腿是否接地（0 或 1）
        #     0.0,    # state[7] 右腿是否接地（0 或 1）
        # ]
        # truncated 强制截断 默认最多 1000 步
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, float(done))
        obs        = next_obs
        ep_reward += reward

        if done:
            obs, _ = env.reset()
            recent_rew.append(ep_reward)
            ep_reward = 0.0
            ep_count += 1

            if ep_count % args.log_interval == 0:
                mean_rew = np.mean(recent_rew)
                print(
                    f"步数 {step:>8,} | 回合 {ep_count:>5} | "
                    f"均分(近20) {mean_rew:>8.1f} | "
                    f"epsilon {epsilon(step):.3f} | "
                    f"loss {np.mean(losses) if losses else 0:.4f}"
                )
                losses.clear()

        # 学习
        if step >= args.learn_start and step % args.learn_freq == 0:
            losses.append(update())

        # 同步目标网络
        if step % args.target_update == 0:
            target_net.load_state_dict(online_net.state_dict())

        # 保存检查点
        if step - last_save >= args.save_interval:
            path = os.path.join(args.save_dir, f"dueling_dqn_{step}.pt")
            torch.save(online_net.state_dict(), path)
            print(f"  → 已保存: {path}")
            last_save = step

    # 最终保存
    final_path = os.path.join(args.save_dir, "dueling_dqn_final.pt")
    torch.save(online_net.state_dict(), final_path)
    print(f"训练完成，模型保存至: {final_path}")
    env.close()


if __name__ == "__main__":
    train()
