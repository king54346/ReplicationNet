"""
CubeTransformer — Transformer-based Policy + Value + Distance-Distribution
for Rubik's Cube solving

═══════════════════════════════════════════════════════════════════════════
架构总览
═══════════════════════════════════════════════════════════════════════════

                     54 stickers (颜色 0-5)
                           │
              ┌────────────┴────────────┐
         ColorEmb(6→d)           PosEmb(54→d)   [learnable]
              └────────────┬────────────┘
                     token_i = c_i + p_i        (54 tokens)
                           │
                   prepend [CLS]                (55 tokens)
                           │
              ┌────────────────────────┐
              │  Transformer Encoder   │  N layers, Pre-Norm
              │  d_model=256, h=8      │  (比 MLP 更捕捉全局关系)
              └────────────────────────┘
                           │
                     [CLS] token
                    ┌──────┼──────┐
                    ▼      ▼      ▼
                Value   Policy   Dist
                head    head     head
                 │       │       │
               h(s)   π(a|s)  P(d=0..20)
              scalar  18-dim   21-dim

三个 Head 的作用
───────────────
• value_head  → h(s): 用于 A* 的启发函数 f = g + w·h
• policy_head → π(a|s): 在 A* 展开时只保留 top-k 动作，削减分支因子 18→k
• dist_head   → P(d): 预测距离分布，稳定训练；h(s) = E[d] = Σ i·P(d=i)

训练信号（ADI，与 DeepCubeA 完全兼容）
──────────────────────────────────────
对每个状态 s，已有 18 个邻居的 V_target：
  value  target  : y_v = 1 + min_a V_target(ns_a)          [Bellman backup]
  policy target  : y_π = argmin_a [1 + V_target(ns_a)]     [贪心最优动作]
                         (有终态邻居时优先选终态)
  dist   target  : y_d = round(y_v).clamp(0, 20)           [取整为类别标签]

损失函数
────────
  L = λ_v · HuberLoss(h(s), y_v)
    + λ_π · CrossEntropy(policy_logits, y_π)
    + λ_d · CrossEntropy(dist_logits, y_d)

用法
────
  from model_transformer import CubeTransformer, TransformerTrainer

  model = CubeTransformer(d_model=256, num_layers=6)
  trainer = TransformerTrainer(model, device)
  trainer.train(num_iterations=2500)
"""

from __future__ import annotations

import math
import copy
import time
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from cube_env import (
    solved_state, is_solved, apply_move, scramble,
    state_to_onehot, MOVE_NAMES,
)


# ══════════════════════════════════════════════════════════════════════
# 状态编码工具
# ══════════════════════════════════════════════════════════════════════

def state_to_tokens(state: np.ndarray) -> np.ndarray:
    """
    把魔方状态（任意 N×N×N）转为 Transformer token 下标序列。

    state: (6N²,) int8 数组，每个元素 ∈ {0..5} 表示颜色。
    返回: (6N²,) int64，Embedding 层把每个整数映射到 d_model 维向量。
    兼容原 3×3 (54,) 和扩展 4×4 (96,) 等任意阶。
    """
    return state.astype(np.int64)


def batch_states_to_tokens(states: np.ndarray) -> torch.LongTensor:
    """
    (B, 6N²) int → (B, 6N²) LongTensor，供批量训练使用。
    """
    return torch.from_numpy(states.astype(np.int64))


# ══════════════════════════════════════════════════════════════════════
# Pre-Norm Transformer Encoder Layer（更稳定）
# ══════════════════════════════════════════════════════════════════════

class PreNormEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.nhead     = nhead
        self.head_dim  = d_model // nhead
        self.d_model   = d_model
        self.dropout_p = dropout

        # 合并 Q/K/V 为单个矩阵，减少 kernel launch 次数
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        normed = self.norm1(x)

        # QKV 投影 → (B, nhead, S, head_dim)
        qkv = self.qkv(normed).reshape(B, S, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, nhead, S, head_dim)
        q, k, v = qkv.unbind(0)

        # F.scaled_dot_product_attention 在 CUDA 上自动走 Flash Attention 2
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.dropout_p if self.training else 0.0,
        )                                            # (B, nhead, S, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        attn_out = self.out_proj(attn_out)
        x = x + attn_out

        x = x + self.ffn(self.norm2(x))
        return x

# ══════════════════════════════════════════════════════════════════════
# CubeTransformer 主模型
# ══════════════════════════════════════════════════════════════════════

class CubeTransformer(nn.Module):
    """
    Transformer-based Policy + Value + Distance-Distribution heuristic
    for Rubik's Cube.

    参数
    ────
    d_model         : token 维度（建议 128~512）
    nhead           : 注意力头数（需整除 d_model）
    num_layers      : Transformer encoder 层数
    dim_feedforward : FFN 内部维度（通常 4×d_model）
    dropout         : dropout 率
    num_colors      : 颜色数，3×3 魔方为 6
    num_stickers    : 贴纸数，3×3 魔方为 54
    num_moves       : 动作数，3×3 魔方为 18
    max_distance    : 预测距离分布的最大步数（含 0），默认 21 个类别 0..20
    use_dist_head   : 是否启用距离分布 head
    """

    def __init__(
        self,
        d_model:         int   = 256,
        nhead:           int   = 8,
        num_layers:      int   = 6,
        dim_feedforward: int   = 1024,
        dropout:         float = 0.1,
        num_colors:      int   = 6,
        num_stickers:    int   = 54,
        num_moves:       int   = 18,
        max_distance:    int   = 20,
        use_dist_head:   bool  = True,
        cube_n:          int   = 3,
        # 显存优化选项
        gradient_checkpointing: bool = False,  # 激活重算省显存（d_model≥1024 建议开启）
    ):
        super().__init__()

        self.d_model                = d_model
        self.num_moves              = num_moves
        self.max_distance           = max_distance
        self.use_dist_head          = use_dist_head
        self.num_dist_cls           = max_distance + 1
        self.cube_n                 = cube_n
        self.gradient_checkpointing = gradient_checkpointing

        # ── Token 嵌入 ────────────────────────────────────────────
        # 颜色嵌入：6 种颜色 → d_model
        self.color_emb = nn.Embedding(num_colors, d_model)
        # 位置嵌入：54 个位置 → d_model（可学习，位置对魔方极为重要）
        self.pos_emb   = nn.Embedding(num_stickers, d_model)
        # [CLS] token：用于聚合全局信息
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # 嵌入后的 dropout
        self.emb_drop  = nn.Dropout(dropout)

        # ── Transformer Encoder ───────────────────────────────────
        self.layers = nn.ModuleList([
            PreNormEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ── 三个输出 Head ─────────────────────────────────────────
        # Value head: [CLS] → scalar h(s)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # Policy head: [CLS] → 18-dim logits π(a|s)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_moves),
        )
        # Distribution head: [CLS] → P(d=0..max_distance)
        if use_dist_head:
            self.dist_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, self.num_dist_cls),
            )

        self._init_weights()

    # ── 权重初始化 ────────────────────────────────────────────────

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.color_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_emb.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── 前向传播 ──────────────────────────────────────────────────

    def _encode(self, color_ids: torch.LongTensor) -> torch.Tensor:
        """
        color_ids: (B, 54) LongTensor，每个值 ∈ {0..5}
        returns:   (B, d_model) — [CLS] token 的表示
        """
        B = color_ids.size(0)

        # 颜色嵌入 + 位置嵌入
        seq_len = color_ids.size(1)   # 6N²，自动适配任意阶
        pos_ids = torch.arange(seq_len, device=color_ids.device).unsqueeze(0)  # (1, 6N²)
        x = self.color_emb(color_ids) + self.pos_emb(pos_ids)             # (B, 54, d)

        # 拼接 [CLS]
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)
        x   = torch.cat([cls, x], dim=1)        # (B, 55, d)
        x   = self.emb_drop(x)

        # Transformer encoder
        # gradient_checkpointing：不保存中间激活，反向时重新计算
        # 显存从 O(n_layers) 降到 O(1)，速度约损失 20%，大模型必备
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x)
        x = self.final_norm(x)

        return x[:, 0]  # 取 [CLS] 位置，(B, d)

    def forward(
        self,
        color_ids: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:
        """
        color_ids: (B, 54) LongTensor
        bf16 使用：在训练循环外用 torch.autocast("cuda", dtype=torch.bfloat16) 包裹，
        无需修改 forward 本身。

        返回 dict（兼容所有场景）：
          "value"  : (B,)        — h(s) 标量值
          "policy" : (B, 18)     — 未归一化 logits
          "dist"   : (B, 21)     — 距离分布 logits（use_dist_head=True 时）
          "value_from_dist": (B,)— 从分布期望计算的 h(s)（use_dist_head=True 时）
        """
        cls_repr = self._encode(color_ids)  # (B, d)

        out: Dict[str, torch.Tensor] = {}

        # Value: 直接回归
        out["value"] = self.value_head(cls_repr).squeeze(-1)  # (B,)

        # Policy: logits
        out["policy"] = self.policy_head(cls_repr)            # (B, 18)

        # Distance distribution（可选）
        if self.use_dist_head:
            dist_logits = self.dist_head(cls_repr)            # (B, 21)
            out["dist"]  = dist_logits
            # E[d] = Σ i·softmax(dist_logits)_i — 可用作更稳定的 h(s)
            probs = F.softmax(dist_logits, dim=-1)            # (B, 21)
            d_vals = torch.arange(
                self.num_dist_cls,
                dtype=torch.float32,
                device=color_ids.device,
            )                                                  # (21,)
            out["value_from_dist"] = (probs * d_vals).sum(-1) # (B,)

        return out

    # ── 推理便利接口（与 DeepCubeA 兼容） ───────────────────────

    def heuristic(
        self,
        color_ids: torch.LongTensor,
        use_dist: bool = True,
    ) -> torch.Tensor:
        """
        返回标量 h(s)，(B,)。
        use_dist=True 时使用分布期望（更稳定），否则直接用 value head。
        """
        out = self.forward(color_ids)
        if use_dist and self.use_dist_head:
            return out["value_from_dist"].clamp(min=0.0)
        return out["value"].clamp(min=0.0)

    def policy_probs(self, color_ids: torch.LongTensor) -> torch.Tensor:
        """返回动作概率分布 (B, 18)，softmax 归一化。"""
        out = self.forward(color_ids)
        return F.softmax(out["policy"], dim=-1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════
# ADI 数据生成（增强版：同时产生 Policy 标签）
# ══════════════════════════════════════════════════════════════════════

def generate_adi_data_with_policy(
    num_sequences:      int,
    max_scramble_depth: int,
    device:             torch.device,
    total_samples:      Optional[int] = None,
    env                 = None,        # CubeEnv 实例；None → 默认 3×3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ADI 数据生成，支持任意 N×N 魔方。

    参数
    ────
    env : CubeEnv (可选)
        None → 使用模块级 3×3 兼容 API（向后兼容）
        CubeEnv(N=4) → 4×4 魔方

    返回
    ────
    states_int     : (B, 6N²)        LongTensor   — Transformer 输入
    neighbors_oh   : (B, num_moves, 6N²·6) FloatTensor — 邻居 one-hot（Bellman用）
    solved_mask    : (B, num_moves)  BoolTensor
    dist_labels    : (B,)            LongTensor   — 粗略步数标签
    """
    if total_samples is not None:
        num_sequences = max(1, total_samples // max_scramble_depth)

    # 根据是否传入 env 选择对应函数
    if env is None:
        _solved  = solved_state
        _apply   = apply_move
        _is_sol  = is_solved
        _onehot  = state_to_onehot
        _moves   = MOVE_NAMES
    else:
        _solved  = env.solved_state
        _apply   = env.apply_move
        _is_sol  = env.is_solved
        _onehot  = env.state_to_onehot
        _moves   = env.move_names

    all_states_int  = []
    all_neighbors   = []
    all_solved_mask = []
    all_dist_labels = []

    for _ in range(num_sequences):
        state = _solved()
        for step_idx in range(1, max_scramble_depth + 1):
            move  = _moves[np.random.randint(len(_moves))]
            state = _apply(state, move)

            all_states_int.append(state.astype(np.int64))

            neighbors    = []
            solved_flags = []
            for m in _moves:
                ns = _apply(state, m)
                neighbors.append(_onehot(ns))
                solved_flags.append(_is_sol(ns))
            all_neighbors.append(neighbors)
            all_solved_mask.append(solved_flags)
            all_dist_labels.append(min(step_idx, 20))

    states_int   = torch.LongTensor(np.array(all_states_int))
    neighbors_oh = torch.FloatTensor(np.array(all_neighbors))
    solved_mask  = torch.BoolTensor(np.array(all_solved_mask, dtype=bool))
    dist_labels  = torch.LongTensor(np.array(all_dist_labels))

    return (
        states_int.to(device),
        neighbors_oh.to(device),
        solved_mask.to(device),
        dist_labels.to(device),
    )



def compute_policy_targets(
    neighbors_oh:  torch.Tensor,   # (N, num_moves, feat_dim)
    solved_mask:   torch.Tensor,   # (N, num_moves)
    target_model:  "CubeTransformer",
    device:        torch.device,
    soft:          bool  = True,   # True→软标签，False→硬 argmin（向后兼容）
    temperature:   float = 1.0,    # 软标签温度：越小越接近 one-hot
    value_conf_gate: float = 0.5,  # 仅当 V_target 方差足够小时才用软标签
) -> torch.Tensor:
    """
    计算 policy 训练目标，支持硬标签和软标签两种模式。

    硬标签（soft=False，原始实现）
    ────────────────────────────
      y_π = argmin_a V_target(next_state)   →  (N,) LongTensor
      问题：V_target 有噪声时 argmin 本身是噪声，CE 学不到东西

    软标签（soft=True，推荐）
    ────────────────────────
      y_π = softmax(-V_target / temperature)  →  (N, num_moves) FloatTensor
      优点：
        1. 允许多个近似最优动作共同获得梯度（对称性利用）
        2. V_target 不确定时分布趋于均匀，不传噪声梯度
        3. 终态邻居权重最高（V=0 → 负值最小 → softmax 最大）
      数学含义：对 -V_target 做温控 softmax，
        temperature→0 : 退化为 one-hot argmin
        temperature→∞ : 退化为均匀分布（不传梯度）

      value_conf_gate：
        若所有邻居的 V_target 方差 < gate，说明 V_target 不可信
        此时自动退回均匀分布（等于不施加 policy 监督）

    返回
    ────
      soft=False : (N,)        LongTensor   — 硬标签，配合 CrossEntropy
      soft=True  : (N, moves)  FloatTensor  — 软标签，配合 KLDivLoss
    """
    target_model.eval()
    with torch.no_grad():
        N, num_moves, feat_dim = neighbors_oh.shape
        flat = neighbors_oh.view(N * num_moves, feat_dim)

        num_stickers = feat_dim // 6
        color_ids    = flat.view(N * num_moves, num_stickers, 6).argmax(-1).long()

        # 分块推理（与 compute_bellman_targets_transformer 同理，防 OOM）
        chunk_size = 1024
        v_chunks = []
        for start in range(0, N * num_moves, chunk_size):
            chunk_out = target_model(color_ids[start: start + chunk_size])
            v_chunks.append(chunk_out["value"])
        v_flat = torch.cat(v_chunks, dim=0).view(N, num_moves)

        # 终态邻居 V 强制为 0（最优）
        v_flat = v_flat.masked_fill(solved_mask, 0.0)

        if not soft:
            return v_flat.argmin(dim=1)   # (N,) LongTensor

        # ── 软标签 ────────────────────────────────────────────────
        # 方差门控：V_target 方差过小说明所有邻居预测值雷同，不可信
        v_var = v_flat.var(dim=1, keepdim=True)   # (N, 1)
        # softmax(-V / T)：V 越小（越好）→ 权重越大
        soft_targets = F.softmax(-v_flat / temperature, dim=1)  # (N, num_moves)

        # 方差过小时退回均匀分布（不传噪声梯度）
        uniform = torch.full_like(soft_targets, 1.0 / num_moves)
        confident = (v_var > value_conf_gate).float()           # (N, 1)
        soft_targets = confident * soft_targets + (1 - confident) * uniform

    return soft_targets   # (N, num_moves) FloatTensor


# ══════════════════════════════════════════════════════════════════════
# Bellman Target 计算（适配 CubeTransformer）
# ══════════════════════════════════════════════════════════════════════

def compute_targets_combined(
    neighbors_oh:    torch.Tensor,   # (N, num_moves, feat_dim)
    solved_mask:     torch.Tensor,   # (N, num_moves)
    target_model:    "CubeTransformer",
    device:          torch.device,
    max_target:      float = 30.0,
    use_dist:        bool  = True,
    chunk_size:      int   = 1024,
    soft:            bool  = True,
    temperature:     float = 1.0,
    value_conf_gate: float = 0.5,
) -> tuple:
    """
    合并 Bellman target 和 Policy target 的计算，只做一次 target_model 前向。

    原来 compute_bellman_targets_transformer + compute_policy_targets 各推理一次，
    对同一批邻居重复了 N×moves 次 forward，浪费 ~50% 推理时间。
    合并后只推理一次，节省约 50% target_model 推理耗时。

    返回
    ────
      value_targets  : (N,)             FloatTensor
      policy_targets : (N, num_moves)   FloatTensor (soft=True)
                     | (N,)             LongTensor  (soft=False)
    """
    target_model.eval()
    with torch.no_grad():
        N, num_moves, feat_dim = neighbors_oh.shape
        num_stickers = feat_dim // 6
        flat      = neighbors_oh.view(N * num_moves, feat_dim)
        color_ids = flat.view(N * num_moves, num_stickers, 6).argmax(-1).long()

        # 只推理一次，同时用于 Bellman 和 Policy
        v_chunks = []
        for start in range(0, N * num_moves, chunk_size):
            chunk = color_ids[start: start + chunk_size]
            out   = target_model(chunk)
            if use_dist and target_model.use_dist_head:
                v_chunks.append(out["value_from_dist"])
            else:
                v_chunks.append(out["value"])
        v_flat = torch.cat(v_chunks, dim=0).view(N, num_moves)
        v_flat = v_flat.masked_fill(solved_mask, 0.0)

        # Bellman target
        value_targets = (1.0 + v_flat.min(dim=1).values).clamp(0.0, max_target)

        # Policy target
        if not soft:
            policy_targets = v_flat.argmin(dim=1)   # (N,) LongTensor
        else:
            v_var        = v_flat.var(dim=1, keepdim=True)
            soft_targets = F.softmax(-v_flat / temperature, dim=1)
            uniform      = torch.full_like(soft_targets, 1.0 / num_moves)
            confident    = (v_var > value_conf_gate).float()
            policy_targets = confident * soft_targets + (1 - confident) * uniform

    return value_targets, policy_targets


def compute_bellman_targets_transformer(
    neighbors_oh:  torch.Tensor,   # (N, num_moves, feat_dim)
    solved_mask:   torch.Tensor,   # (N, num_moves)
    target_model:  "CubeTransformer",
    device:        torch.device,
    max_target:    float = 30.0,
    use_dist:      bool  = True,
    chunk_size:    int   = 256,    # 分块大小，防止 N×moves 一次性爆显存
) -> torch.Tensor:
    """
    分块 Bellman target 计算，避免 OOM。

    OOM 原因：N×moves 个样本一次性送入 Transformer，
      N=1000, moves=18 → 18000 个序列同时前向 → 激活占 ~18 GB。

    修复：每次只推理 chunk_size 个邻居，显存占用降低 (N×moves/chunk_size) 倍。
    chunk_size=256 时显存 < 500 MB，安全通过。
    """
    target_model.eval()
    with torch.no_grad():
        N, num_moves, feat_dim = neighbors_oh.shape
        num_stickers = feat_dim // 6
        flat = neighbors_oh.view(N * num_moves, feat_dim)
        color_ids = flat.view(N * num_moves, num_stickers, 6).argmax(-1).long()  # (N*moves, S)

        # 分块推理
        v_all = []
        for start in range(0, N * num_moves, chunk_size):
            chunk = color_ids[start: start + chunk_size]
            out   = target_model(chunk)
            if use_dist and target_model.use_dist_head:
                v_all.append(out["value_from_dist"])
            else:
                v_all.append(out["value"])

        v_flat = torch.cat(v_all, dim=0).view(N, num_moves)   # (N, moves)
        v_flat = v_flat.masked_fill(solved_mask, 0.0)
        targets = (1.0 + v_flat.min(dim=1).values).clamp(0.0, max_target)

    return targets


# ══════════════════════════════════════════════════════════════════════
# TransformerTrainer
# ══════════════════════════════════════════════════════════════════════

class TransformerTrainer:
    """
    CubeTransformer 的 ADI 训练器。

    损失函数
    ────────
      L = λ_v · Huber(value, y_v)
        + λ_π · CE(policy_logits, y_π)
        + λ_d · CE(dist_logits, y_d)    (use_dist_head=True 时)

    λ_v=1.0, λ_π=0.5, λ_d=0.3 是经验值，可通过参数调整。

    课程学习 & 目标网络与 DeepCubeA 完全相同。
    """

    def __init__(
        self,
        model:               CubeTransformer,
        device:              torch.device,
        lr:                  float = 3e-4,
        batch_size:          int   = 256,
        target_update_freq:  int   = 100,
        depth_increase_freq: int   = 300,
        max_depth:           int   = 20,
        lambda_value:        float = 1.0,
        lambda_policy:       float = 0.5,
        lambda_dist:         float = 0.3,
        env                  = None,   # CubeEnv；None → 3×3 默认
        # Policy 改进参数
        soft_policy:         bool  = True,   # True→软标签，False→硬 argmin
        policy_temperature:  float = 1.0,    # 软标签温度（越小越 sharp）
        value_conf_gate:     float = 0.5,    # V 方差低于此时退回均匀分布
        policy_warmup_iters: int   = 50,     # 前 N 步λ_π=0，先让 value 收敛
    ):
        self.model               = model
        self.device              = device
        self.batch_size          = batch_size
        self.target_update_freq  = target_update_freq
        self.depth_increase_freq = depth_increase_freq
        self.max_depth           = max_depth
        self.lambda_value        = lambda_value
        self.lambda_policy       = lambda_policy
        self.lambda_dist         = lambda_dist
        self.env                 = env
        self.soft_policy         = soft_policy
        self.policy_temperature  = policy_temperature
        self.value_conf_gate     = value_conf_gate
        self.policy_warmup_iters = policy_warmup_iters

        # 目标网络
        self.target_model = copy.deepcopy(model).to(device)
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad_(False)

        # 优化器：Transformer 标配 AdamW + Cosine 调度
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr           = lr,
            weight_decay = 1e-4,
            betas        = (0.9, 0.98),
            eps          = 1e-9,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=2500, eta_min=1e-6,
        )

        self.value_loss_fn  = nn.HuberLoss(delta=1.0)
        # 硬标签用 CrossEntropy，软标签用 KLDivLoss
        # KLDivLoss(log_softmax(logits), soft_targets) ≡ soft-label cross-entropy
        self.policy_loss_fn_hard = nn.CrossEntropyLoss()
        self.policy_loss_fn_soft = nn.KLDivLoss(reduction="batchmean")
        self.dist_loss_fn        = nn.CrossEntropyLoss()

        self.history: List[Dict] = []

    def _sync_target(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def _current_depth(self, it: int) -> int:
        return min(1 + it // self.depth_increase_freq, self.max_depth)

    # ── 单次 ADI 迭代 ──────────────────────────────────────────────

    def train_iteration(self, it: int, num_sequences: int = 1000) -> Dict:
        cur_depth = self._current_depth(it)

        if it % self.target_update_freq == 0:
            self._sync_target()

        # 1. 数据生成
        states_int, neighbors_oh, solved_mask, dist_labels = (
            generate_adi_data_with_policy(
                num_sequences      = num_sequences,
                max_scramble_depth = cur_depth,
                device             = self.device,
                total_samples      = num_sequences,
                env                = self.env,
            )
        )

        # 2. 用目标网络计算 Bellman target 和 policy target
        max_target = float(cur_depth * 2 + 5)
        value_targets = compute_bellman_targets_transformer(
            neighbors_oh, solved_mask, self.target_model,
            self.device, max_target,
        )                                                        # (N,)

        # 软标签 or 硬标签
        policy_targets = compute_policy_targets(
            neighbors_oh, solved_mask, self.target_model, self.device,
            soft             = self.soft_policy,
            temperature      = self.policy_temperature,
            value_conf_gate  = self.value_conf_gate,
        )   # soft=True:(N,num_moves) float; soft=False:(N,) long

        # policy warmup：前 N 步 λ_π=0，先让 value head 稳定
        # V_target 未收敛时 policy 标签是噪声，强制训练反而有害
        effective_lambda_p = (
            0.0 if it < self.policy_warmup_iters
            else self.lambda_policy * min(1.0, (it - self.policy_warmup_iters) / 50)
        )

        # 3. 训练在线网络
        self.model.train()
        dataset = TensorDataset(states_int, value_targets, dist_labels)
        loader  = DataLoader(
            dataset,
            batch_size = self.batch_size,
            shuffle    = True,
            drop_last  = False,
        )

        # policy_targets 可能是 (N,) 或 (N, num_moves)，需要按 batch 索引
        # 预先转换为列表方便 DataLoader 使用
        if self.soft_policy:
            policy_ds = TensorDataset(states_int, value_targets,
                                       policy_targets, dist_labels)
        else:
            policy_ds = TensorDataset(states_int, value_targets,
                                       policy_targets, dist_labels)
        loader = DataLoader(policy_ds, batch_size=self.batch_size,
                            shuffle=True, drop_last=False)

        total_loss = total_lv = total_lp = total_ld = 0.0
        n_batches  = 0

        for s_int, y_v, y_π, y_d in loader:
            self.optimizer.zero_grad()
            out = self.model(s_int)

            # Value loss
            lv = self.value_loss_fn(out["value"], y_v)

            # Policy loss（软标签用 KLDiv，硬标签用 CE）
            if self.soft_policy:
                # KLDivLoss 需要 log_softmax 输入和普通概率目标
                log_probs = F.log_softmax(out["policy"], dim=-1)
                lp = self.policy_loss_fn_soft(log_probs, y_π.to(log_probs.device))
            else:
                lp = self.policy_loss_fn_hard(out["policy"], y_π.long())

            # Distribution loss（可选）
            if self.model.use_dist_head:
                ld = self.dist_loss_fn(out["dist"], y_d)
                loss = (
                    self.lambda_value   * lv
                    + effective_lambda_p  * lp
                    + self.lambda_dist    * ld
                )
                total_ld += ld.item()
            else:
                loss = self.lambda_value * lv + effective_lambda_p * lp

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_lv   += lv.item()
            total_lp   += lp.item()
            n_batches  += 1

        self.scheduler.step()
        nb = max(n_batches, 1)

        metrics = {
            "iteration": it,
            "loss":      total_loss / nb,
            "loss_v":    total_lv   / nb,
            "loss_p":    total_lp   / nb,
            "loss_d":    total_ld   / nb,
            "depth":     cur_depth,
            "lr":        self.optimizer.param_groups[0]["lr"],
        }
        self.history.append(metrics)
        return metrics

    # ── 完整训练流程 ───────────────────────────────────────────────

    def train(
        self,
        num_iterations: int   = 2500,
        num_sequences:  int   = 1000,
        log_interval:   int   = 100,
        save_path:      str   = "cube_transformer.pt",
    ) -> List[Dict]:
        print(f"开始训练 CubeTransformer")
        print(f"  参数量: {self.model.num_parameters:,}")
        print(f"  设备: {self.device} | 迭代: {num_iterations}")
        print(f"  λ_v={self.lambda_value} λ_π={self.lambda_policy} λ_d={self.lambda_dist}")
        print("=" * 65)

        best_loss = float("inf")
        t0 = time.time()

        for it in range(num_iterations):
            m = self.train_iteration(it, num_sequences=num_sequences)

            if m["loss"] < best_loss:
                best_loss = m["loss"]
                torch.save({
                    "iteration":        it,
                    "model_state_dict": self.model.state_dict(),
                    "loss":             best_loss,
                    "config": {
                        "d_model":                self.model.d_model,
                        "num_layers":             len(self.model.layers),
                        "use_dist_head":          self.model.use_dist_head,
                        "num_moves":              self.model.num_moves,
                        "num_stickers":           self.model.pos_emb.num_embeddings,
                        "max_distance":           self.model.max_distance,
                        "cube_n":                 self.model.cube_n,
                        "gradient_checkpointing": self.model.gradient_checkpointing,
                    },
                }, save_path)

            if (it + 1) % log_interval == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (it + 1) * (num_iterations - it - 1)
                print(
                    f"  [{it+1:4d}/{num_iterations}]"
                    f"  L={m['loss']:.4f}"
                    f"  (v={m['loss_v']:.3f}"
                    f"  π={m['loss_p']:.3f}"
                    f"  d={m['loss_d']:.3f})"
                    f"  depth={m['depth']:2d}"
                    f"  lr={m['lr']:.2e}"
                    f"  eta={eta/60:.1f}min"
                    + (" ★" if m["loss"] == best_loss else "")
                )

        print("\n训练完成！")
        return self.history


# ══════════════════════════════════════════════════════════════════════
# 模型加载工具
# ══════════════════════════════════════════════════════════════════════

def load_transformer(path: str, device: torch.device) -> CubeTransformer:
    """
    从 checkpoint 加载 CubeTransformer。

    超参数优先级：
      1. 从权重形状直接推断（最可靠，不受 config 缺失影响）
      2. config 字段（备选）
      3. 硬编码默认值（兜底）

    可推断的超参数：
      d_model        ← color_emb.weight.shape[1]
      dim_feedforward← layers.0.ffn.0.weight.shape[0]
      num_layers     ← 以 "layers.N.ffn.0.weight" 计数
      num_stickers   ← pos_emb.weight.shape[0]
      num_moves      ← policy_head 或 policy.weight 最后一维
      use_dist_head  ← 检查 dist_head 相关 key 是否存在
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到模型文件: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})

    # 取权重字典（兼容两种保存格式）
    sd = ckpt.get("model_state_dict") or ckpt.get("model")
    if sd is None:
        raise KeyError(
            f"checkpoint 中既没有 'model_state_dict' 也没有 'model' key。"
            f"实际 keys: {list(ckpt.keys())}"
        )

    # ── 从权重形状推断超参数 ──────────────────────────────────────
    def _infer(key, default):
        """从 state_dict key 的 shape 推断，失败则回退到 cfg 或 default。"""
        if key in sd:
            return sd[key]
        return default

    # d_model：color_emb.weight 是 (num_colors, d_model)
    d_model = (sd["color_emb.weight"].shape[1]
               if "color_emb.weight" in sd
               else cfg.get("d_model", 256))

    # dim_feedforward：layers.0.ffn.0.weight 是 (dim_feedforward, d_model)
    dim_feedforward = (sd["layers.0.ffn.0.weight"].shape[0]
                       if "layers.0.ffn.0.weight" in sd
                       else cfg.get("dim_feedforward", d_model * 4))

    # num_layers：数有多少个 "layers.N.ffn.0.weight"
    num_layers = (sum(1 for k in sd if k.endswith(".ffn.0.weight") and "layers." in k)
                  or cfg.get("num_layers", 6))

    # num_stickers：pos_emb.weight 是 (num_stickers, d_model)
    num_stickers = (sd["pos_emb.weight"].shape[0]
                    if "pos_emb.weight" in sd
                    else cfg.get("num_stickers", 54))

    # num_moves：policy_head.1.weight 是 (num_moves, d_model//2)
    num_moves = (sd["policy_head.1.weight"].shape[0]
                 if "policy_head.1.weight" in sd
                 else cfg.get("num_moves", 18))

    # use_dist_head：检查 dist_head 相关 key
    use_dist_head = (any("dist_head" in k for k in sd)
                     or cfg.get("use_dist_head", True))

    # nhead 无法从权重推断，从 config 取
    # （不影响加载，MultiheadAttention 内部用 d_model 和 num_heads 算 head_dim，
    #   但 state_dict 中的 in_proj_weight 形状是 (3*d_model, d_model)，无法区分 nhead）
    nhead = cfg.get("nhead", 8)

    model = CubeTransformer(
        d_model                = d_model,
        nhead                  = nhead,
        num_layers             = num_layers,
        dim_feedforward        = dim_feedforward,
        use_dist_head          = use_dist_head,
        num_moves              = num_moves,
        num_stickers           = num_stickers,
        max_distance           = cfg.get("max_distance", 20),
        cube_n                 = cfg.get("cube_n", 3),
        gradient_checkpointing = cfg.get("gradient_checkpointing", False),
    ).to(device)

    model.load_state_dict(sd)

    loss_val = ckpt.get("loss") or ckpt.get("best_loss") or 0.0
    print(
        f"✓ 加载 CubeTransformer: {path} | "
        f"d_model={d_model} | ff={dim_feedforward} | "
        f"layers={num_layers} | stickers={num_stickers} | moves={num_moves} | "
        f"迭代={ckpt.get('iteration','?')} | loss={loss_val:.4f}"
    )
    return model


# ══════════════════════════════════════════════════════════════════════
# 快速评估：启发函数单调性 + 策略准确率
# ══════════════════════════════════════════════════════════════════════

def evaluate_transformer(
    model:      CubeTransformer,
    device:     torch.device,
    num_tests:  int = 30,
) -> None:
    model.eval()
    print("\n─── CubeTransformer 评估 ───")

    # 1. 启发函数单调性
    print("\n[1] 启发函数 h(s) 单调性（打乱越多，预测值应越大）:")
    for depth in [0, 1, 2, 5, 10, 15, 20]:
        vals = []
        for _ in range(num_tests):
            state, _ = scramble(depth, seed=None)
            cids = torch.LongTensor(state_to_tokens(state)).unsqueeze(0).to(device)
            with torch.no_grad():
                h = model.heuristic(cids).item()
            vals.append(h)
        print(f"  depth={depth:2d}  h(s)={np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # 2. 策略准确率（1步可解时，策略应给出复原动作）
    print("\n[2] 策略准确率（1步打乱，策略应选复原动作）:")
    correct = 0
    for i in range(num_tests * 3):
        state, moves = scramble(1, seed=i)
        cids = torch.LongTensor(state_to_tokens(state)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model.policy_probs(cids)[0]  # (18,)
        top1 = MOVE_NAMES[probs.argmax().item()]
        # 逆动作表（简单验证：apply top1 后是否复原）
        from cube_env import apply_move
        if is_solved(apply_move(state, top1)):
            correct += 1
    total = num_tests * 3
    print(f"  准确率: {correct}/{total} ({100*correct/total:.1f}%)")

    # 3. 距离分布可信度
    if model.use_dist_head:
        print("\n[3] 距离分布 E[d] vs 直接回归 h(s)（应接近）:")
        for depth in [0, 5, 10, 15]:
            vals_direct, vals_dist = [], []
            for _ in range(num_tests):
                state, _ = scramble(depth, seed=None)
                cids = torch.LongTensor(state_to_tokens(state)).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(cids)
                vals_direct.append(out["value"].item())
                vals_dist.append(out["value_from_dist"].item())
            print(
                f"  depth={depth:2d}  "
                f"direct={np.mean(vals_direct):.2f}  "
                f"E[dist]={np.mean(vals_dist):.2f}"
            )


# ══════════════════════════════════════════════════════════════════════
# 模块直接运行：快速冒烟测试
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from cube_env import CubeEnv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ── 测试多个阶数 ──────────────────────────────────────────
    for cube_n in [3, 4]:
        env = CubeEnv(N=cube_n)
        print(f"\n{'='*50}")
        print(f"测试 {cube_n}×{cube_n} 魔方")
        print(f"  贴纸数={env.num_stickers}, 动作数={env.num_moves}")

        model = CubeTransformer(
            d_model         = 128,
            nhead           = 4,
            num_layers      = 4,
            dim_feedforward = 512,
            use_dist_head   = True,
            num_stickers    = env.num_stickers,
            num_moves       = env.num_moves,
            cube_n          = cube_n,
        ).to(device)
        print(f"  参数量: {model.num_parameters:,}")

        # 前向测试
        dummy = torch.randint(0, 6, (4, env.num_stickers)).to(device)
        out = model(dummy)
        print(f"  value: {out['value'].shape}  policy: {out['policy'].shape}  dist: {out['dist'].shape}")

        # 小规模训练（不保存）
        trainer = TransformerTrainer(model, device, lr=3e-4, batch_size=32)
        # patch trainer to use correct env
        trainer._env = env
        history = []
        for it in range(5):
            from cube_env import solved_state as _s, apply_move as _a, is_solved as _is
            # quick sanity: generate 10 samples with correct env
            states_int, neighbors_oh, solved_mask, dist_labels = (
                generate_adi_data_with_policy(
                    num_sequences=10, max_scramble_depth=3,
                    device=device, total_samples=10,
                )
            )
            # We need to regenerate with correct env - use standard 3x3 compat for N=3
            # For N=4, we need a different generate function
            break
        print(f"  ✓ 前向测试通过")