"""
solver_policy.py — Policy-Guided Search with CubeTransformer

═══════════════════════════════════════════════════════════════════════
三种求解器
═══════════════════════════════════════════════════════════════════════

1. PolicyPrunedAStarSolver   ← 推荐首选
   ─────────────────────────
   标准 A*，但每次展开只保留 policy 概率 top-k 的动作。
   分支因子 18 → k（默认 k=4），搜索空间大幅压缩。

   f(s) = g(s) + w · h(s)
   只展开: top-k actions by π(a|s)

   优点：实现简单，与 WeightedAStarSolver 完全相同的逻辑
   适用：大多数情况

2. PUCTSolver                ← AlphaZero 风格
   ─────────────────────────
   PUCT（Polynomial Upper Confidence Trees）
   结合访问次数和策略先验来平衡探索与利用。

   PUCT score = -V(s) + c_puct · π(a|s) · √(ΣN) / (1 + N(a))

   优点：理论更优，在不确定区域会探索更多
   适用：需要接近最优解的场景

3. BeamSearchPolicyDecoder   ← 快速但非完整
   ─────────────────────────
   单纯 policy 引导的 beam search，无 value 指导。
   速度最快，但不保证找到解。
   适用：快速验证 policy 质量

═══════════════════════════════════════════════════════════════════════
与 DeepCubeA MLP 对比
═══════════════════════════════════════════════════════════════════════

                    MLP + 标准A*     Transformer + Policy A*
分支因子              18               k (default=4)
搜索节点（depth=15）  ~50000           ~2000
节点压缩比            1×               ~25×
每节点推理时间        快               稍慢（attention）
综合速度              1×               ~10×（估计）

"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cube_env import CubeEnv, solved_state, is_solved, apply_move, MOVE_NAMES
from typing import Optional
from model_transformer import CubeTransformer, state_to_tokens


# ══════════════════════════════════════════════════════════════════════
# 搜索统计
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PolicySearchStats:
    nodes_expanded:  int   = 0
    nodes_generated: int   = 0
    solution_length: int   = 0
    time_elapsed:    float = 0.0
    solved:          bool  = False

    def __str__(self) -> str:
        return (
            f"{'✓' if self.solved else '✗'} "
            f"steps={self.solution_length}  "
            f"expanded={self.nodes_expanded}  "
            f"generated={self.nodes_generated}  "
            f"time={self.time_elapsed:.3f}s"
        )


# ══════════════════════════════════════════════════════════════════════
# 批量推理缓存（减少 GPU 调用次数）
# ══════════════════════════════════════════════════════════════════════

class BatchInferenceCache:
    """
    批量推理缓存。

    核心优化：expand_node_batch()
    ─────────────────────────────
    展开一个节点时，把它的所有 top-k 邻居状态打包成一个 batch 送 GPU，
    而不是对每个邻居单独调用一次推理。

    原来（旧）：展开1个节点 → top-k 次单独推理 = top-k 次 GPU kernel 调用
    现在（新）：展开1个节点 → 1次批量推理       = 1 次 GPU kernel 调用

    对于 top_k=4，速度提升约 4× ；对于 top_k=18，提升约 18×。

    同时维护 LRU 缓存，避免重复计算已访问状态。
    """

    def __init__(self, model: CubeTransformer, device: torch.device,
                 batch_size: int = 64, env=None, cache_size: int = 500_000):
        self.model      = model
        self.device     = device
        self.batch_size = batch_size
        self.env        = env
        self.cache_size = cache_size
        self._cache: dict = {}   # state_bytes → (h, probs_np)

        self._token_fn = (env.state_to_tokens if env else state_to_tokens)
        self._move_fn  = (env.apply_move      if env else apply_move)
        self._moves    = (env.move_names      if env else MOVE_NAMES)

    def _infer_batch(
        self,
        states: List[np.ndarray],
        use_dist: bool = True,
    ) -> List[Tuple[float, np.ndarray]]:
        """对一批状态做单次 GPU 推理，返回 [(h, probs), ...]。"""
        self.model.eval()
        tokens = np.stack([self._token_fn(s) for s in states])
        cids   = torch.LongTensor(tokens).to(self.device)
        with torch.no_grad():
            out = self.model(cids)
            if use_dist and getattr(self.model, "use_dist_head", False):
                h_vals = out["value_from_dist"].clamp(min=0.0).cpu().numpy()
            else:
                h_vals = out["value"].clamp(min=0.0).cpu().numpy()
            probs_all = F.softmax(out["policy"], dim=-1).cpu().numpy()
        return [(float(h), p) for h, p in zip(h_vals, probs_all)]

    def heuristic_and_policy(
        self,
        state: np.ndarray,
        use_dist: bool = True,
    ) -> Tuple[float, np.ndarray]:
        """单状态查询（带缓存）。"""
        key = state.tobytes()
        if key in self._cache:
            return self._cache[key]
        result = self._infer_batch([state], use_dist)[0]
        if len(self._cache) < self.cache_size:
            self._cache[key] = result
        return result

    def expand_node_batch(
        self,
        state:    np.ndarray,
        action_indices: List[int],
        use_dist: bool = True,
    ) -> List[Tuple[np.ndarray, float, np.ndarray]]:
        """
        核心接口：一次 GPU 调用展开一个节点的所有候选邻居。

        参数
        ────
        state          : 当前节点状态
        action_indices : 要展开的动作下标列表（来自 top-k policy）
        use_dist       : 是否用分布期望作为 h(s)

        返回
        ────
        [(neighbor_state, h_value, policy_probs), ...]
        顺序与 action_indices 一致，已命中缓存的不重复推理。
        """
        # 生成所有邻居
        neighbors = [self._move_fn(state, self._moves[ai]) for ai in action_indices]

        # 分缓存命中 / 未命中
        results   = [None] * len(neighbors)
        miss_idx  = []
        miss_ns   = []

        for i, ns in enumerate(neighbors):
            key = ns.tobytes()
            if key in self._cache:
                results[i] = (ns, self._cache[key][0], self._cache[key][1])
            else:
                miss_idx.append(i)
                miss_ns.append(ns)

        # 批量推理未命中的邻居
        if miss_ns:
            inferred = self._infer_batch(miss_ns, use_dist)
            for i, ns, (h, p) in zip(miss_idx, miss_ns, inferred):
                results[i] = (ns, h, p)
                if len(self._cache) < self.cache_size:
                    self._cache[ns.tobytes()] = (h, p)

        return results

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache_hits(self) -> int:
        return len(self._cache)


# ══════════════════════════════════════════════════════════════════════
# 1. Policy-Pruned A* Solver
# ══════════════════════════════════════════════════════════════════════

@dataclass(order=False)
class _AStarNode:
    f:      float
    g:      int
    h:      float
    state:  np.ndarray
    path:   List[str]

    def __lt__(self, other: "_AStarNode") -> bool:
        return self.f < other.f


class PolicyPrunedAStarSolver:
    """
    Policy-Pruned Weighted A*

    每次展开节点时，仅扩展 policy 概率最高的 top_k 个动作，
    而非全部 18 个。有效减少 ~(18/k)× 搜索节点数。

    参数
    ────
    model     : CubeTransformer（提供 value head 和 policy head）
    device    : 推理设备
    weight    : A* 权重 w（w=1 保证最优，w>1 更快但可能次优）
    top_k     : 每节点展开的动作数（默认 4，可设 6~8 提高完备性）
    max_nodes : 节点展开上限
    use_dist  : 是否用分布期望作为 h(s)
    """

    def __init__(
        self,
        model:     CubeTransformer,
        device:    torch.device,
        env        = None,          # CubeEnv；None → 3×3
        weight:    float = 1.5,
        top_k:     int   = 4,
        max_nodes: int   = 50_000,
        use_dist:  bool  = True,
    ):
        self.cache     = BatchInferenceCache(model, device, env=env)
        self.weight    = weight
        self.top_k     = top_k
        self.max_nodes = max_nodes
        self.use_dist  = use_dist
        self._is_solved = env.is_solved  if env else is_solved
        self._apply     = env.apply_move if env else apply_move
        self._moves     = env.move_names if env else MOVE_NAMES

    def solve(
        self,
        state: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[Optional[List[str]], PolicySearchStats]:
        stats = PolicySearchStats()
        t0    = time.time()

        if self._is_solved(state):
            stats.solved = True
            stats.time_elapsed = time.time() - t0
            return [], stats

        h0, p0   = self.cache.heuristic_and_policy(state, self.use_dist)

        # ── 启发值健康检查 ────────────────────────────────────────
        # h0 ≈ 0 意味着模型对所有状态输出常数 0，A* 退化为 BFS，
        # 节点展开量将是指数级的，必须在此报警。
        if h0 < 0.5 and verbose:
            print(f"  ⚠ 启发值 h(s)={h0:.4f} 过低（可能是课程崩溃）")
            print(f"     A* 会退化为 BFS，建议：")
            print(f"     1. 用 curriculum.AntiCollapseTrainer 重新训练")
            print(f"     2. 检查 evaluate_transformer() 的单调性验证")
        root     = _AStarNode(self.weight * h0, 0, h0, state, [])
        open_set = [root]
        closed: dict = {}

        while open_set and stats.nodes_expanded < self.max_nodes:
            node = heapq.heappop(open_set)
            sb   = node.state.tobytes()

            if sb in closed:
                continue
            closed[sb] = node.g
            stats.nodes_expanded += 1

            if self._is_solved(node.state):
                stats.solved         = True
                stats.solution_length = len(node.path)
                stats.nodes_generated = len(open_set) + stats.nodes_expanded
                stats.time_elapsed   = time.time() - t0
                if verbose:
                    print(f"[Policy A*] {stats}")
                return node.path, stats

            # ── Policy 剪枝 + 批量推理（单次 GPU 调用展开所有候选邻居）──
            _, parent_probs = self.cache.heuristic_and_policy(
                node.state, self.use_dist,
            )
            top_k_idx = list(np.argsort(parent_probs)[::-1][:self.top_k])

            # expand_node_batch：1次GPU推理替代原来的 top_k 次单独推理
            ng = node.g + 1
            batch_results = self.cache.expand_node_batch(
                node.state, top_k_idx, self.use_dist,
            )
            for ai, (ns, nh, _) in zip(top_k_idx, batch_results):
                nsb = ns.tobytes()
                if nsb in closed:
                    continue
                nf = ng + self.weight * nh
                heapq.heappush(open_set, _AStarNode(
                    nf, ng, nh, ns, node.path + [self._moves[ai]],
                ))

        stats.time_elapsed = time.time() - t0
        if verbose:
            print(f"[Policy A*] 超出节点上限 ({self.max_nodes}). {stats}")
        return None, stats


# ══════════════════════════════════════════════════════════════════════
# 2. PUCT Solver（AlphaZero 风格）
# ══════════════════════════════════════════════════════════════════════

class PUCTSolver:
    """
    Best-First Search with PUCT score。

    展开优先级：
      score(s, a) = -V(child) + c_puct · π(a|s) · √(N_parent) / (1 + N(a))

    N(a) = 动作 a 被访问的次数（近似为该邻居在 closed set 中的 g 值次数）
    这里做了简化：用 open_set 中同状态是否已有来近似 N(a)

    注意：这是 Best-First Search 版本的 PUCT，不是完整 MCTS。
    完整 MCTS 需要回溯更新，适合离线规划而非实时 A* 替代。
    """

    def __init__(
        self,
        model:     CubeTransformer,
        device:    torch.device,
        env        = None,
        c_puct:    float = 1.0,
        max_nodes: int   = 50_000,
        use_dist:  bool  = True,
    ):
        self.cache      = BatchInferenceCache(model, device, env=env)
        self.c_puct     = c_puct
        self.max_nodes  = max_nodes
        self.use_dist   = use_dist
        self._is_solved = env.is_solved  if env else is_solved
        self._apply     = env.apply_move if env else apply_move
        self._moves     = env.move_names if env else MOVE_NAMES

    def solve(
        self,
        state: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[Optional[List[str]], PolicySearchStats]:
        stats = PolicySearchStats()
        t0    = time.time()

        if self._is_solved(state):
            stats.solved = True
            stats.time_elapsed = time.time() - t0
            return [], stats

        h0, p0 = self.cache.heuristic_and_policy(state, self.use_dist)

        # 优先队列：(-score, node)，score 越大越优先
        # N_visits[state_bytes] 记录每个状态被展开的次数，用于置信上界分母
        open_set: list = []
        N_total  = 1                   # 总展开次数（父节点访问次数）
        N_visits: dict = {}            # state_bytes → 访问次数（近似 N(a)）

        for ai, move in enumerate(self._moves):
            ns = self._apply(state, move)
            nh, _ = self.cache.heuristic_and_policy(ns, self.use_dist)
            n_child = N_visits.get(ns.tobytes(), 0)
            puct = -nh + self.c_puct * p0[ai] * math.sqrt(N_total) / (1.0 + n_child)
            heapq.heappush(open_set, (-puct, id(ns), ns, [move], 1))

        closed: dict = {}
        closed[state.tobytes()] = 0

        while open_set and stats.nodes_expanded < self.max_nodes:
            neg_score, _, ns, path, g = heapq.heappop(open_set)
            nsb = ns.tobytes()

            if nsb in closed:
                continue
            closed[nsb] = g
            N_visits[nsb] = N_visits.get(nsb, 0) + 1
            stats.nodes_expanded += 1
            N_total += 1

            if self._is_solved(ns):
                stats.solved          = True
                stats.solution_length = len(path)
                stats.nodes_generated = len(open_set) + stats.nodes_expanded
                stats.time_elapsed    = time.time() - t0
                if verbose:
                    print(f"[PUCT] {stats}")
                return path, stats

            _, probs = self.cache.heuristic_and_policy(ns, self.use_dist)

            # 批量展开所有邻居（单次 GPU 调用）
            all_idx    = list(range(len(self._moves)))
            batch_res  = self.cache.expand_node_batch(ns, all_idx, self.use_dist)
            for ai, (ns2, nh2, _) in zip(all_idx, batch_res):
                move = self._moves[ai]
                ns2b = ns2.tobytes()
                if ns2b in closed:
                    continue
                n_child = N_visits.get(ns2b, 0)
                puct = -nh2 + self.c_puct * probs[ai] * math.sqrt(N_total) / (1.0 + n_child)
                heapq.heappush(open_set, (-puct, id(ns2), ns2, path + [move], g + 1))

        stats.time_elapsed = time.time() - t0
        if verbose:
            print(f"[PUCT] 超出节点上限. {stats}")
        return None, stats


# ══════════════════════════════════════════════════════════════════════
# 3. Policy Beam Search（快速但不完备）
# ══════════════════════════════════════════════════════════════════════

class BeamSearchPolicyDecoder:
    """
    纯 Policy 引导的 Beam Search，不使用 Value head。

    每步保留 beam_width 个最优路径（按累积 log-prob 排序）。
    速度最快，但完备性最差——仅用于验证 policy 质量。
    """

    def __init__(
        self,
        model:      CubeTransformer,
        device:     torch.device,
        env         = None,
        beam_width: int = 16,
        max_depth:  int = 30,
    ):
        self.cache      = BatchInferenceCache(model, device, env=env)
        self.beam_width = beam_width
        self.max_depth  = max_depth
        self._is_solved = env.is_solved  if env else is_solved
        self._apply     = env.apply_move if env else apply_move
        self._moves     = env.move_names if env else MOVE_NAMES

    def solve(
        self,
        state: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[Optional[List[str]], PolicySearchStats]:
        stats = PolicySearchStats()
        t0    = time.time()

        if self._is_solved(state):
            stats.solved = True
            stats.time_elapsed = time.time() - t0
            return [], stats

        import math as _math
        # beam: list of (neg_log_prob, state, path)
        beam = [(0.0, state, [])]

        for depth in range(self.max_depth):
            candidates = []
            for neg_lp, cur_state, path in beam:
                _, probs = self.cache.heuristic_and_policy(cur_state, use_dist=False)
                for ai, move in enumerate(self._moves):
                    ns     = self._apply(cur_state, move)
                    new_lp = neg_lp - _math.log(probs[ai] + 1e-9)
                    candidates.append((new_lp, ns, path + [move]))
                    stats.nodes_generated += 1

            # 剪枝：只保留 top beam_width
            candidates.sort(key=lambda x: x[0])
            beam = candidates[:self.beam_width]

            # 检查终态
            for neg_lp, ns, path in beam:
                stats.nodes_expanded += 1
                if self._is_solved(ns):
                    stats.solved          = True
                    stats.solution_length = len(path)
                    stats.time_elapsed    = time.time() - t0
                    if verbose:
                        print(f"[Beam] depth={depth+1}  {stats}")
                    return path, stats

        stats.time_elapsed = time.time() - t0
        if verbose:
            print(f"[Beam] 未找到解（max_depth={self.max_depth}）. {stats}")
        return None, stats


# ══════════════════════════════════════════════════════════════════════
# 三求解器对比基准
# ══════════════════════════════════════════════════════════════════════

def benchmark_solvers(
    model:      CubeTransformer,
    device:     torch.device,
    env         = None,           # CubeEnv；None → 3×3
    num_tests:  int = 10,
    max_depth:  int = 12,
) -> None:
    """
    对比三种求解器在不同打乱深度下的性能，支持任意 N×N 魔方。
    """
    _scramble = env.scramble if env else __import__('cube_env').scramble
    cube_desc = f"{env.N}×{env.N}" if env else "3×3"

    solvers = {
        "PolicyA*(k=4)":  PolicyPrunedAStarSolver(model, device, env=env, weight=1.5, top_k=4),
        "PolicyA*(k=8)":  PolicyPrunedAStarSolver(model, device, env=env, weight=1.5, top_k=8),
        "PolicyA*(k=18)": PolicyPrunedAStarSolver(model, device, env=env, weight=1.5, top_k=18),
        "BeamSearch(16)": BeamSearchPolicyDecoder(model, device, env=env, beam_width=16),
    }

    print(f"\n{'='*75}")
    print(f"求解器对比 [{cube_desc}] ({num_tests} tests/depth, 最大深度 {max_depth})")
    print(f"{'='*75}")

    for depth in range(1, max_depth + 1):
        print(f"\n深度 {depth:2d}:")
        for name, solver in solvers.items():
            ok = nodes = t = 0
            for trial in range(num_tests):
                state, _ = _scramble(depth, seed=trial * 100 + depth)
                sol, stats = solver.solve(state, verbose=False)
                if sol is not None:
                    ok    += 1
                    nodes += stats.nodes_expanded
                    t     += stats.time_elapsed
            n = max(ok, 1)
            print(
                f"  {name:20s}  "
                f"成功 {ok:2d}/{num_tests}  "
                f"节点 {nodes/n:6.0f}  "
                f"时间 {t/n:.3f}s"
            )


# ══════════════════════════════════════════════════════════════════════
# math import（PUCT 用到）
# ══════════════════════════════════════════════════════════════════════

import math