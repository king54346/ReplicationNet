"""
model.py — DeepCubeA 神经网络 + ADI 训练
支持任意 N×N×N 魔方（通过 CubeEnv 传入）

网络结构（与原论文一致）
─────────────────────────
  Input: 6N² × 6 = 36N² 维 one-hot
    ↓  Linear + BN + ELU
  Hidden: hidden_dim 维残差块 × num_blocks
    ↓  Linear
  Output: 标量 V(s)，估计距复原步数

训练算法：ADI（Autodidactic Iteration）
─────────────────────────────────────────
  1. 从复原态随机游走生成轨迹
  2. 对每个状态计算 18/36/… 个邻居的 V_target
  3. Bellman backup: y(s) = 1 + min_a V_target(next(s,a))
  4. MSE/Huber loss 训练在线网络
  5. 周期性将在线网络同步到目标网络

多阶使用
─────────
  # 3×3（默认，向后兼容）
  model = DeepCubeA()
  trainer = Trainer(model, device)

  # 4×4
  from cube_env import CubeEnv
  env = CubeEnv(N=4)
  model = DeepCubeA.for_env(env)
  trainer = Trainer(model, device, env=env)
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from cube_env import (
    CubeEnv,
    solved_state, is_solved, apply_move, scramble,
    state_to_onehot, MOVE_NAMES,
)


# ══════════════════════════════════════════════════════════════════════
# 网络模块
# ══════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """BN-ELU 残差块（与原论文一致）。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DeepCubeA(nn.Module):
    """
    DeepCubeA 启发函数网络，支持任意 N×N×N 魔方。

    输入: 36N² 维 one-hot（6N² 格 × 6 色）
    输出: 标量 V(s)，估计距复原的步数

    参数
    ────
    input_dim  : one-hot 维度，默认 324（3×3）；4×4 应传 576
    hidden_dim : 隐藏层宽度，原论文 4096，演示用 512
    num_blocks : 残差块数量，原论文 4
    cube_n     : 魔方阶数，仅用于 checkpoint 存档
    """

    def __init__(
        self,
        input_dim:  int = 324,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        cube_n:     int = 3,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.cube_n     = cube_n

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) → (B,)"""
        x = self.input_proj(x)
        for blk in self.res_blocks:
            x = blk(x)
        return self.head(x).squeeze(-1)

    @classmethod
    def for_env(
        cls,
        env:        CubeEnv,
        hidden_dim: int = 512,
        num_blocks: int = 4,
    ) -> "DeepCubeA":
        """根据 CubeEnv 自动设置 input_dim 和 cube_n。"""
        return cls(
            input_dim  = env.onehot_size,
            hidden_dim = hidden_dim,
            num_blocks = num_blocks,
            cube_n     = env.N,
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════
# ADI 数据生成
# ══════════════════════════════════════════════════════════════════════

def generate_adi_data(
    num_sequences:      int,
    max_scramble_depth: int,
    device:             torch.device,
    total_samples:      Optional[int] = None,
    env:                Optional[CubeEnv] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ADI 数据生成：从复原态随机游走。

    参数
    ────
    env : CubeEnv（可选）
        None → 使用 3×3 兼容 API（向后兼容）
        CubeEnv(N=4) → 4×4 魔方数据

    返回
    ────
    states_oh      : (B, 36N²)         FloatTensor  — 状态 one-hot
    neighbors_oh   : (B, num_moves, 36N²) FloatTensor  — 邻居 one-hot
    solved_mask    : (B, num_moves)    BoolTensor   — 邻居是否复原
    """
    if total_samples is not None:
        num_sequences = max(1, total_samples // max_scramble_depth)

    # 选择环境函数
    if env is None:
        _solved = solved_state
        _apply  = apply_move
        _is_sol = is_solved
        _onehot = state_to_onehot
        _moves  = MOVE_NAMES
    else:
        _solved = env.solved_state
        _apply  = env.apply_move
        _is_sol = env.is_solved
        _onehot = env.state_to_onehot
        _moves  = env.move_names

    all_states    = []
    all_neighbors = []
    all_solved    = []

    for _ in range(num_sequences):
        state = _solved()
        for _ in range(max_scramble_depth):
            move  = _moves[np.random.randint(len(_moves))]
            state = _apply(state, move)

            all_states.append(_onehot(state))

            nbrs, flags = [], []
            for m in _moves:
                ns = _apply(state, m)
                nbrs.append(_onehot(ns))
                flags.append(_is_sol(ns))
            all_neighbors.append(nbrs)
            all_solved.append(flags)

    return (
        torch.FloatTensor(np.array(all_states)).to(device),
        torch.FloatTensor(np.array(all_neighbors)).to(device),
        torch.BoolTensor(np.array(all_solved, dtype=bool)).to(device),
    )


# ══════════════════════════════════════════════════════════════════════
# Bellman 目标计算
# ══════════════════════════════════════════════════════════════════════

def compute_bellman_targets(
    neighbors_oh:  torch.Tensor,    # (B, num_moves, 36N²)
    solved_mask:   torch.Tensor,    # (B, num_moves)
    target_model:  DeepCubeA,
    device:        torch.device,
    max_target:    float = 30.0,
) -> torch.Tensor:
    """
    y(s) = min_a [1 + V_target(next(s,a))]
    终态邻居的 V 强制为 0。
    """
    target_model.eval()
    with torch.no_grad():
        B, num_moves, feat = neighbors_oh.shape
        flat      = neighbors_oh.view(B * num_moves, feat)
        v_flat    = target_model(flat)
        v_neigh   = v_flat.view(B, num_moves)
        v_neigh   = v_neigh.masked_fill(solved_mask, 0.0)
        targets   = (1.0 + v_neigh.min(dim=1).values).clamp(0.0, max_target)
    return targets


# ══════════════════════════════════════════════════════════════════════
# 评估工具
# ══════════════════════════════════════════════════════════════════════

def evaluate_heuristic(
    model:     DeepCubeA,
    device:    torch.device,
    env:       Optional[CubeEnv] = None,
    num_tests: int = 20,
) -> None:
    """验证启发函数单调性：打乱越多，预测值应越大。"""
    model.eval()
    _scramble = env.scramble if env else scramble
    _onehot   = env.state_to_onehot if env else state_to_onehot
    cube_desc = f"{env.N}×{env.N}" if env else "3×3"

    print(f"\n{cube_desc} 启发函数单调性验证:")
    print("-" * 40)
    prev_avg = -1.0
    monotone = True
    for depth in [0, 1, 2, 5, 10, 15, 20]:
        vals = []
        for _ in range(num_tests):
            state, _ = _scramble(depth)
            oh = torch.FloatTensor(_onehot(state)).unsqueeze(0).to(device)
            with torch.no_grad():
                vals.append(model(oh).item())
        avg = float(np.mean(vals))
        print(f"  打乱 {depth:2d} 步 → {avg:.3f}")
        if avg < prev_avg - 0.05:
            monotone = False
        prev_avg = avg
    print(f"  单调性: {'✓ 通过' if monotone else '✗ 未通过（需继续训练）'}")


# ══════════════════════════════════════════════════════════════════════
# 训练器
# ══════════════════════════════════════════════════════════════════════

class Trainer:
    """
    DeepCubeA ADI 训练器，支持任意 N×N×N 魔方。

    参数
    ────
    model                : DeepCubeA 实例
    device               : 训练设备
    env                  : CubeEnv 实例；None → 3×3 默认
    lr                   : Adam 学习率
    batch_size           : mini-batch 大小
    target_update_freq   : 目标网络同步间隔（迭代数）
    depth_increase_freq  : 课程学习深度增加间隔（迭代数）
    max_depth            : 课程学习最大打乱深度
    """

    def __init__(
        self,
        model:               DeepCubeA,
        device:              torch.device,
        env:                 Optional[CubeEnv] = None,
        lr:                  float = 1e-3,
        batch_size:          int   = 512,
        target_update_freq:  int   = 100,
        depth_increase_freq: int   = 300,
        max_depth:           int   = 20,
    ):
        self.model               = model
        self.device              = device
        self.env                 = env
        self.batch_size          = batch_size
        self.target_update_freq  = target_update_freq
        self.depth_increase_freq = depth_increase_freq
        self.max_depth           = max_depth

        # 目标网络
        self.target_model = copy.deepcopy(model).to(device)
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=2500, eta_min=1e-5,
        )
        self.loss_fn = nn.HuberLoss(delta=1.0)
        self.history: List[Dict] = []

    def _sync_target(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def _current_depth(self, it: int) -> int:
        return min(1 + it // self.depth_increase_freq, self.max_depth)

    def train_iteration(self, it: int, num_sequences: int = 1000) -> float:
        cur_depth = self._current_depth(it)

        if it % self.target_update_freq == 0:
            self._sync_target()

        # 数据生成
        states_oh, neighbors_oh, solved_mask = generate_adi_data(
            num_sequences      = num_sequences,
            max_scramble_depth = cur_depth,
            device             = self.device,
            total_samples      = num_sequences,
            env                = self.env,
        )

        # Bellman 目标
        targets = compute_bellman_targets(
            neighbors_oh, solved_mask, self.target_model, self.device,
            max_target = float(cur_depth * 2 + 5),
        )

        # 训练
        self.model.train()
        loader = DataLoader(
            TensorDataset(states_oh, targets),
            batch_size = self.batch_size,
            shuffle    = True,
            drop_last  = False,
        )
        total_loss, n_batches = 0.0, 0
        for s_b, t_b in loader:
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(s_b), t_b)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        self.scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        self.history.append({
            'iteration': it, 'loss': avg_loss,
            'depth': cur_depth, 'lr': self.optimizer.param_groups[0]['lr'],
        })
        return avg_loss

    def train(
        self,
        num_iterations: int = 500,
        num_sequences:  int = 1000,
        log_interval:   int = 50,
        save_path:      str = 'deepcubea.pt',
    ) -> List[Dict]:
        cube_desc = f"{self.env.N}×{self.env.N}" if self.env else "3×3"
        print(f"开始训练 DeepCubeA ({cube_desc}魔方)")
        print(f"  input_dim={self.model.input_dim}  hidden={self.model.hidden_dim}"
              f"  blocks={len(self.model.res_blocks)}  参数量={self.model.num_parameters:,}")
        print(f"  设备={self.device}  迭代={num_iterations}  batch={self.batch_size}")
        print("=" * 60)

        best_loss = float('inf')
        t0 = time.time()

        for it in range(num_iterations):
            loss = self.train_iteration(it, num_sequences=num_sequences)

            if loss < best_loss:
                best_loss = loss
                torch.save({
                    'iteration':        it,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state':  self.optimizer.state_dict(),
                    'loss':             loss,
                    'config': {
                        'input_dim':  self.model.input_dim,
                        'hidden_dim': self.model.hidden_dim,
                        'num_blocks': len(self.model.res_blocks),
                        'cube_n':     self.model.cube_n,
                    },
                }, save_path)

            if (it + 1) % log_interval == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (it + 1) * (num_iterations - it - 1)
                h       = self.history[-1]
                print(
                    f"  [{it+1:4d}/{num_iterations}]"
                    f"  loss={loss:.4f}"
                    f"  best={best_loss:.4f}"
                    f"  depth={h['depth']:2d}"
                    f"  lr={h['lr']:.2e}"
                    f"  eta={eta/60:.1f}min"
                    + (" ★" if loss == best_loss else "")
                )

        print(f"\n训练完成！最优 loss: {best_loss:.4f}  已保存: {save_path}")
        return self.history


# ══════════════════════════════════════════════════════════════════════
# 模型加载工具
# ══════════════════════════════════════════════════════════════════════

def load_deepcubea(
    path:       str,
    device:     torch.device,
    hidden_dim: int = 512,
    num_blocks: int = 4,
) -> DeepCubeA:
    """
    从 checkpoint 加载 DeepCubeA，自动推断网络结构（兼容多阶）。
    """
    if not __import__('os').path.exists(path):
        print(f"⚠ 模型文件不存在: {path}，使用随机初始化")
        return DeepCubeA(hidden_dim=hidden_dim, num_blocks=num_blocks)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    # 兼容两种保存格式（单 GPU: "model_state_dict"，分布式 worker: "model"）
    sd = ckpt.get('model_state_dict') or ckpt.get('model')
    if sd is None:
        raise KeyError(f"checkpoint 缺少权重 key，实际 keys: {list(ckpt.keys())}")
    cfg  = ckpt.get('config', {})

    # 从 checkpoint config 或权重形状自动推断
    inferred_input  = cfg.get('input_dim',
        sd['input_proj.0.weight'].shape[1])
    inferred_hidden = cfg.get('hidden_dim',
        sd['input_proj.0.weight'].shape[0])
    inferred_blocks = cfg.get('num_blocks',
        sum(1 for k in sd if k.startswith('res_blocks.') and k.endswith('.net.0.weight')))
    inferred_n      = cfg.get('cube_n', 3)

    model = DeepCubeA(
        input_dim  = inferred_input,
        hidden_dim = inferred_hidden,
        num_blocks = inferred_blocks,
        cube_n     = inferred_n,
    ).to(device)
    model.load_state_dict(sd)

    print(f"✓ 加载 DeepCubeA: {path}")
    print(f"  {inferred_n}×{inferred_n} | input={inferred_input} | "
          f"hidden={inferred_hidden} | blocks={inferred_blocks} | "
          f"迭代={ckpt.get('iteration','?')} | loss={ckpt.get('loss',0):.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════
# 模块直接运行：快速冒烟测试
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    for n in [3, 4]:
        env = CubeEnv(N=n) if n != 3 else None
        print(f"\n{'='*50}")
        print(f"测试 {n}×{n} 魔方")

        model = (DeepCubeA.for_env(CubeEnv(N=n), hidden_dim=128)
                 if n != 3
                 else DeepCubeA(hidden_dim=128, num_blocks=2))
        model = model.to(device)
        print(f"  参数量: {model.num_parameters:,}")

        trainer = Trainer(model, device, env=env,
                          batch_size=64, depth_increase_freq=100)
        trainer.train(num_iterations=20, num_sequences=100,
                      log_interval=10,
                      save_path=f'deepcubea_{n}x{n}_test.pt')

        evaluate_heuristic(model, device, env=env, num_tests=5)