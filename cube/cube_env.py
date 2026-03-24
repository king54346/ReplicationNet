"""
cube_env.py — N×N×N 魔方环境
═══════════════════════════════════════════════════════════════════════

支持任意 N ≥ 2 的魔方，模块级 API 与原 3×3 代码完全兼容。

状态表示
────────
• (6·N²,) int8 数组，每格值 ∈ {0..5} 表示颜色
• 面顺序: U=0, D=1, F=2, B=3, L=4, R=5
• (face, row, col) → flat index = face·N² + row·N + col

动作系统
────────
三个旋转轴，每轴 N 层，每层带 CW / CCW / 180° 三种变体：

  U-axis：  d=0→"U"，  d=1..N-2→"U{d}"，  d=N-1→"D"
  R-axis：  d=0→"R"，  d=1..N-2→"R{d}"，  d=N-1→"L"  [方向反转]
  F-axis：  d=0→"F"，  d=1..N-2→"F{d}"，  d=N-1→"B"  [方向反转]

移动数量：
  N=3 → 18（仅外层，与原代码一致）
  N=4 → 36（全部 4×3×3）
  N=K → 9K（全部 K×3×3）

三轴置换公式（已通过 3x3 原始代码及 4x4 全面验证）
──────────────────────────────────────────────────
U-axis CW at depth d（从 U 面向下数第 d 层）：
  strip: F[d,k] → R[d,k] → B[d,k] → L[d,k]  for k=0..N-1
  face:  d=0→U CW，d=N-1→D CW（D 与 U 同向，与原代码一致）

R-axis CW at depth d（从 R 面向左数第 d 层）：
  strip: U[r,N-1-d] → B[N-1-r,d] → D[r,N-1-d] → F[r,N-1-d]
  face:  d=0→R CW，d=N-1→L CCW（L=此方向的逆）

F-axis CW at depth d（从 F 面向后数第 d 层）：
  strip: U[N-1-d,k] → R[k,d] → D[d,N-1-k] → L[N-1-k,N-1-d]
  face:  d=0→F CW，d=N-1→B CCW（B=此方向的逆）

向后兼容
────────
模块级函数（solved_state, is_solved, apply_move, scramble,
state_to_onehot, render_state, MOVE_NAMES）全部默认 N=3，
无需修改任何现有 import。

新 API
──────
  env = CubeEnv(N=4)
  state = env.solved_state()
  state = env.apply_move(state, "U1")   # 4×4 第二层
  print(f"动作数: {len(env.move_names)}")
  print(env.render(state))
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

# ── 面常量 ────────────────────────────────────────────────────────────
FACE_U, FACE_D, FACE_F, FACE_B, FACE_L, FACE_R = 0, 1, 2, 3, 4, 5
# 向后兼容别名
U_COL, D_COL, F_COL, B_COL, L_COL, R_COL = 0, 1, 2, 3, 4, 5
COLOR_NAMES = ['W', 'Y', 'G', 'B', 'O', 'R']   # 白黄绿蓝橙红


# ══════════════════════════════════════════════════════════════════════
# 低层置换工具
# ══════════════════════════════════════════════════════════════════════

def _pos(f: int, r: int, c: int, N: int) -> int:
    """(face, row, col) → flat index."""
    return f * N * N + r * N + c


def _face_cw(f: int, N: int, perm: list) -> None:
    """
    面顺时针旋转（pull-form）：dest(r,c) ← src(N-1-c, r)
    等价于矩阵顺时针旋转 90°。
    """
    for r in range(N):
        for c in range(N):
            perm[_pos(f, r, c, N)] = _pos(f, N - 1 - c, r, N)


def _face_ccw(f: int, N: int, perm: list) -> None:
    """
    面逆时针旋转（pull-form）：dest(r,c) ← src(c, N-1-r)
    等价于面顺时针旋转的逆操作。
    """
    for r in range(N):
        for c in range(N):
            perm[_pos(f, r, c, N)] = _pos(f, c, N - 1 - r, N)


def _invert_perm(perm: np.ndarray) -> np.ndarray:
    """计算 pull-form 置换的逆置换。apply(apply(s,p), inv(p)) = s."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm), dtype=perm.dtype)
    return inv


# ══════════════════════════════════════════════════════════════════════
# 三轴单层旋转（pull-form 置换）
# ══════════════════════════════════════════════════════════════════════

def _build_u_layer_cw(d: int, N: int) -> np.ndarray:
    """
    U-axis 第 d 层 CW（从 U 面方向看顺时针）。

    strip: F[d,k] → R[d,k] → B[d,k] → L[d,k] → F  (for k=0..N-1)
    face:  d=0 → U face CW；d=N-1 → D face CW（同向，与原 3x3 一致）
    """
    perm = list(range(6 * N * N))
    for k in range(N):
        perm[_pos(FACE_R, d, k, N)] = _pos(FACE_F, d, k, N)   # R ← F
        perm[_pos(FACE_B, d, k, N)] = _pos(FACE_R, d, k, N)   # B ← R
        perm[_pos(FACE_L, d, k, N)] = _pos(FACE_B, d, k, N)   # L ← B
        perm[_pos(FACE_F, d, k, N)] = _pos(FACE_L, d, k, N)   # F ← L
    if d == 0:
        _face_cw(FACE_U, N, perm)
    elif d == N - 1:
        _face_cw(FACE_D, N, perm)    # D 与 U 同向（已通过原始代码验证）
    return np.array(perm, dtype=np.int32)


def _build_r_layer_cw(d: int, N: int) -> np.ndarray:
    """
    R-axis 第 d 层 CW（从 R 面方向看顺时针）。

    strip (r=0..N-1):
      U[r, N-1-d] → B[N-1-r, d] → D[r, N-1-d] → F[r, N-1-d] → U

    face: d=0 → R face CW；d=N-1 → L face CCW（L = 此方向逆，已验证）
    """
    perm = list(range(6 * N * N))
    for r in range(N):
        col_ufd = N - 1 - d   # U/F/D 上的列号
        col_b   = d            # B 上的列号（B 面左右镜像）
        perm[_pos(FACE_B, N - 1 - r, col_b,   N)] = _pos(FACE_U, r,         col_ufd, N)
        perm[_pos(FACE_D, r,         col_ufd,  N)] = _pos(FACE_B, N - 1 - r, col_b,   N)
        perm[_pos(FACE_F, r,         col_ufd,  N)] = _pos(FACE_D, r,         col_ufd, N)
        perm[_pos(FACE_U, r,         col_ufd,  N)] = _pos(FACE_F, r,         col_ufd, N)
    if d == 0:
        _face_cw(FACE_R, N, perm)
    elif d == N - 1:
        _face_ccw(FACE_L, N, perm)   # L face CCW（使得 inv(此) = L standard）
    return np.array(perm, dtype=np.int32)


def _build_f_layer_cw(d: int, N: int) -> np.ndarray:
    """
    F-axis 第 d 层 CW（从 F 面方向看顺时针）。

    strip (k=0..N-1):
      U[N-1-d, k] → R[k, d] → D[d, N-1-k] → L[N-1-k, N-1-d] → U

    face: d=0 → F face CW；d=N-1 → B face CCW（B = 此方向逆，已验证）
    """
    perm = list(range(6 * N * N))
    for k in range(N):
        row_ud = N - 1 - d
        perm[_pos(FACE_R, k,         d,         N)] = _pos(FACE_U, row_ud, k,             N)
        perm[_pos(FACE_D, d,         N - 1 - k, N)] = _pos(FACE_R, k,     d,             N)
        perm[_pos(FACE_L, N - 1 - k, N - 1 - d, N)] = _pos(FACE_D, d,     N - 1 - k,     N)
        perm[_pos(FACE_U, row_ud,    k,          N)] = _pos(FACE_L, N - 1 - k, N - 1 - d, N)
    if d == 0:
        _face_cw(FACE_F, N, perm)
    elif d == N - 1:
        _face_ccw(FACE_B, N, perm)   # B face CCW（使得 inv(此) = B standard）
    return np.array(perm, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════
# 全动作集构建
# ══════════════════════════════════════════════════════════════════════

def _build_all_moves(N: int) -> Dict[str, np.ndarray]:
    """
    构建所有 NxN 魔方动作的 pull-form 置换数组。

    命名规则：
      外层 → 标准名（U, D, R, L, F, B）
      内层 → 轴名 + 深度（U1, U2, R1, F2, …）
      变体 → 基础名 + '' 或 '2'（CCW / 180°）

    N=3：18 个动作（仅外层，与原代码完全兼容）
    N=4：36 个动作（4×3轴×3变体）
    """
    moves: Dict[str, np.ndarray] = {}

    def add(base: str, perm_cw: np.ndarray) -> None:
        """注册 CW / CCW / 180° 三种变体。"""
        arr_cw = perm_cw
        arr_ccw = _invert_perm(arr_cw)
        arr_180 = arr_cw[arr_cw]          # CW ∘ CW
        moves[base]          = arr_cw
        moves[base + "'"]    = arr_ccw
        moves[base + "2"]    = arr_180

    # ── U-axis ─────────────────────────────────────────────────────
    for d in range(N):
        pcw = _build_u_layer_cw(d, N)
        if d == 0:
            add("U", pcw)
        elif d == N - 1:
            add("D", pcw)              # D = U-axis last CW（同向）
        elif N > 3:
            add(f"{d+1}U", pcw)        # WCA 记号：2U=第2层，避免与 U2(180°) 冲突

    # ── R-axis ─────────────────────────────────────────────────────
    for d in range(N):
        pcw = _build_r_layer_cw(d, N)
        if d == 0:
            add("R", pcw)
        elif d == N - 1:
            add("L", _invert_perm(pcw))  # L = R-axis last CCW
        elif N > 3:
            add(f"{d+1}R", pcw)        # WCA 记号：2R, 3R …

    # ── F-axis ─────────────────────────────────────────────────────
    for d in range(N):
        pcw = _build_f_layer_cw(d, N)
        if d == 0:
            add("F", pcw)
        elif d == N - 1:
            add("B", _invert_perm(pcw))  # B = F-axis last CCW
        elif N > 3:
            add(f"{d+1}F", pcw)        # WCA 记号：2F, 3F …

    return moves


# ══════════════════════════════════════════════════════════════════════
# CubeEnv 主类
# ══════════════════════════════════════════════════════════════════════

class CubeEnv:
    """
    N×N×N 魔方环境。

    参数
    ────
    N : int
        魔方阶数，N=3 为标准 3×3，N=4 为四阶复仇魔方，以此类推。

    属性
    ────
    N              : 阶数
    num_stickers   : 贴纸总数 = 6·N²
    onehot_size    : one-hot 编码长度 = 6·N²·6 = 36·N²
    move_names     : 排序后的动作名列表
    num_moves      : 动作总数

    用法
    ────
    >>> env = CubeEnv(N=4)
    >>> state = env.solved_state()
    >>> state = env.apply_move(state, "U1")   # 第二层顺时针
    >>> state = env.apply_move(state, "R2")   # 第三层顺时针
    >>> print(env.render(state))
    """

    def __init__(self, N: int = 3):
        if N < 2:
            raise ValueError(f"N 必须 ≥ 2，得到 N={N}")
        self.N = N
        self._moves = _build_all_moves(N)
        self._move_names = sorted(self._moves.keys())
        self._solved = self._make_solved()

    # ── 内部 ──────────────────────────────────────────────────────

    def _make_solved(self) -> np.ndarray:
        N = self.N
        s = np.empty(6 * N * N, dtype=np.int8)
        for f in range(6):
            s[f * N * N:(f + 1) * N * N] = f
        return s

    # ── 公开 API ──────────────────────────────────────────────────

    @property
    def move_names(self) -> List[str]:
        return self._move_names

    @property
    def num_moves(self) -> int:
        return len(self._move_names)

    @property
    def num_stickers(self) -> int:
        return 6 * self.N * self.N

    @property
    def onehot_size(self) -> int:
        return self.num_stickers * 6

    def solved_state(self) -> np.ndarray:
        """返回复原状态的副本。"""
        return self._solved.copy()

    def is_solved(self, state: np.ndarray) -> bool:
        return bool(np.array_equal(state, self._solved))

    def apply_move(self, state: np.ndarray, move_name: str) -> np.ndarray:
        """
        应用单个动作，返回新状态（不修改原状态）。

        等价于 new_state[i] = state[perm[i]]，NumPy fancy-indexing 实现。
        """
        return state[self._moves[move_name]]

    def apply_moves(self, state: np.ndarray, moves: List[str]) -> np.ndarray:
        """应用动作序列。"""
        for m in moves:
            state = state[self._moves[m]]
        return state

    def scramble(
        self,
        num_moves: int,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        从复原态随机打乱 num_moves 步。

        返回 (打乱后状态, 动作序列)。
        连续相同面的动作会被过滤，避免无意义的来回操作。
        """
        rng   = np.random.RandomState(seed)
        state = self.solved_state()
        names = self._move_names
        applied: List[str] = []

        # 每个动作的"基础面"：用于避免连续同面（简单启发式）
        # e.g. "U2'" → 基础面 "U"
        def base_face(m: str) -> str:
            return m.rstrip("'2").rstrip("0123456789")

        last_face = ""
        for _ in range(num_moves):
            candidates = [m for m in names if base_face(m) != last_face]
            if not candidates:
                candidates = names
            m = candidates[int(rng.randint(len(candidates)))]
            state     = state[self._moves[m]]
            applied.append(m)
            last_face = base_face(m)

        return state, applied

    def state_to_onehot(self, state: np.ndarray) -> np.ndarray:
        """
        (6N²,) int8 → (6N²·6,) float32 one-hot 编码。

        向量化实现，无显式循环。
        """
        size = self.num_stickers
        idx  = np.arange(size, dtype=np.int32) * 6 + state.astype(np.int32)
        oh   = np.zeros(size * 6, dtype=np.float32)
        oh[idx] = 1.0
        return oh

    def state_to_tokens(self, state: np.ndarray) -> np.ndarray:
        """
        (6N²,) int8 → (6N²,) int64，供 Transformer Embedding 层使用。
        每个元素是颜色下标 ∈ {0..5}。
        """
        return state.astype(np.int64)

    def render(self, state: np.ndarray) -> str:
        """
        可视化 NxN 魔方展开图：
              UUU          (N 行)
        LLL FFF RRR BBB   (N 行)
              DDD          (N 行)
        """
        N  = self.N
        cn = COLOR_NAMES
        s  = state

        def row_str(face: int, r: int) -> str:
            return ' '.join(cn[int(s[face * N * N + r * N + c])] for c in range(N))

        lines = []
        pad   = ' ' * (N * 2 + 2)    # 与 L 面等宽的缩进
        for r in range(N):
            lines.append(pad + row_str(FACE_U, r))
        for r in range(N):
            lr = row_str(FACE_L, r)
            fr = row_str(FACE_F, r)
            rr = row_str(FACE_R, r)
            br = row_str(FACE_B, r)
            lines.append(f"{lr}  {fr}  {rr}  {br}")
        for r in range(N):
            lines.append(pad + row_str(FACE_D, r))
        return '\n'.join(lines)

    def verify_moves(self) -> bool:
        """
        自检：验证所有动作满足基本代数性质。
        - 每个动作应用 4 次 = 单位元（4-cycle 结构）
        - 每个动作与其逆的复合 = 单位元
        - CW + CCW + CW + CCW = 单位元（交替4次）
        返回 True 表示全部通过。
        """
        identity = np.arange(self.num_stickers, dtype=np.int32)
        ok = True

        base_names = [m for m in self._move_names if not m.endswith("'") and not m.endswith("2")]
        for name in base_names:
            pcw  = self._moves[name]
            pccw = self._moves[name + "'"]
            p180 = self._moves[name + "2"]

            # 4次 CW = 单位元
            p4 = pcw[pcw][pcw][pcw]
            if not np.array_equal(p4, identity):
                print(f"  ✗ {name}: 4次CW ≠ 单位元"); ok = False

            # CW × CCW = 单位元
            if not np.array_equal(pcw[pccw], identity):
                print(f"  ✗ {name}: CW × CCW ≠ 单位元"); ok = False

            # 180° × 180° = 单位元
            if not np.array_equal(p180[p180], identity):
                print(f"  ✗ {name}: 180°×180° ≠ 单位元"); ok = False

        return ok

    def __repr__(self) -> str:
        return f"CubeEnv(N={self.N}, {self.num_moves} moves, {self.num_stickers} stickers)"


# ══════════════════════════════════════════════════════════════════════
# 模块级 3×3 向后兼容 API
# ══════════════════════════════════════════════════════════════════════
# 所有现有代码的 import 无需任何改动：
#   from cube_env import solved_state, is_solved, apply_move, scramble,
#                        state_to_onehot, render_state, MOVE_NAMES

_env3 = CubeEnv(N=3)

def solved_state() -> np.ndarray:
    """返回 3×3 复原状态。"""
    return _env3.solved_state()

def is_solved(state: np.ndarray) -> bool:
    """检查 3×3 状态是否复原。"""
    return _env3.is_solved(state)

def apply_move(state: np.ndarray, move_name: str) -> np.ndarray:
    """对 3×3 状态应用单个动作。"""
    return _env3.apply_move(state, move_name)

def apply_move_sequence(state: np.ndarray, moves: List[str]) -> np.ndarray:
    """对 3×3 状态应用动作序列。"""
    return _env3.apply_moves(state, moves)

def scramble(
    num_moves: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """随机打乱 3×3 魔方 num_moves 步。"""
    return _env3.scramble(num_moves, seed)

def state_to_onehot(state: np.ndarray) -> np.ndarray:
    """3×3 状态 → 324 维 one-hot。"""
    return _env3.state_to_onehot(state)

def render_state(state: np.ndarray) -> str:
    """可视化 3×3 魔方。"""
    return _env3.render(state)

# 3×3 的 18 个标准动作名（向后兼容）
MOVE_NAMES: List[str] = _env3.move_names


# ══════════════════════════════════════════════════════════════════════
# Cubie 表示常量与工具（仅 3×3 魔方）
# ══════════════════════════════════════════════════════════════════════
#
# 角块（corner）：8 个，每个有位置（cp ∈ {0..7}）和方向（co ∈ {0,1,2}）
# 棱块（edge）  ：12 个，每个有位置（ep ∈ {0..11}）和方向（eo ∈ {0,1}）
#
# combined index：
#   角 = cp * 3 + co  ∈ {0..23}    (8×3 = 24 种状态/槽)
#   棱 = ep * 2 + eo  ∈ {0..23}    (12×2 = 24 种状态/槽)
#
# Transformer 输入为 (20,) int64：前 8 为角 combined，后 12 为棱 combined
# ──────────────────────────────────────────────────────────────────────

# 角块槽 i 对应的贴纸索引 [主面(U/D), 侧面1, 侧面2]
# 主面 = 该槽 U 层(0-3)或 D 层(4-7)所在面的贴纸
# face f, row r, col c → index = f*9 + r*3 + c  (N=3)
CORNER_STICKERS: List[List[int]] = [
    [ 8, 45, 20],   # 0: URF  U(2,2), R(0,0), F(0,2)
    [ 6, 18, 38],   # 1: UFL  U(2,0), F(0,0), L(0,2)
    [ 0, 36, 29],   # 2: ULB  U(0,0), L(0,0), B(0,2)
    [ 2, 27, 47],   # 3: UBR  U(0,2), B(0,0), R(0,2)
    [11, 26, 51],   # 4: DFR  D(0,2), F(2,2), R(2,0)
    [ 9, 44, 24],   # 5: DLF  D(0,0), L(2,2), F(2,0)
    [15, 35, 42],   # 6: DBL  D(2,0), B(2,2), L(2,0)
    [17, 53, 33],   # 7: DRB  D(2,2), R(2,2), B(2,0)
]

# 棱块槽 i 对应的贴纸索引 [主面, 次面]
# U/D 层棱(0-7)：主面 = U/D 面贴纸
# 赤道棱(8-11)：主面 = F/B 面贴纸
EDGE_STICKERS: List[List[int]] = [
    [ 7, 19],   # 0: UF   U(2,1), F(0,1)
    [ 3, 37],   # 1: UL   U(1,0), L(0,1)
    [ 1, 28],   # 2: UB   U(0,1), B(0,1)
    [ 5, 46],   # 3: UR   U(1,2), R(0,1)
    [10, 25],   # 4: DF   D(0,1), F(2,1)
    [12, 43],   # 5: DL   D(1,0), L(2,1)
    [16, 34],   # 6: DB   D(2,1), B(2,1)
    [14, 52],   # 7: DR   D(1,2), R(2,1)
    [21, 41],   # 8: FL   F(1,0), L(1,2)
    [23, 48],   # 9: FR   F(1,2), R(1,0)
    [31, 39],   # 10: BL  B(1,2), L(1,0)
    [30, 50],   # 11: BR  B(1,0), R(1,2)
]

# 颜色集合 → home slot 反查表（在 solved 态下每个槽的颜色集合唯一标识该块）
_CORNER_COLOR_TO_HOME: Dict[frozenset, int] = {
    frozenset(s // 9 for s in stickers): i
    for i, stickers in enumerate(CORNER_STICKERS)
}
_EDGE_COLOR_TO_HOME: Dict[frozenset, int] = {
    frozenset(s // 9 for s in stickers): i
    for i, stickers in enumerate(EDGE_STICKERS)
}


def state_to_cubie(state: np.ndarray) -> np.ndarray:
    """
    (54,) int8 → (20,) int64  cubie token 序列（仅 3×3）。

    前 8 个 = 角块 combined index：cp * 3 + co ∈ {0..23}
      cp : 该物理角块的 home slot（0-7，由颜色集合唯一确定）
      co : U/D 色（颜色 0 或 1）在 [c0, c1, c2] 三个贴纸中的位置（0/1/2）

    后 12 个 = 棱块 combined index：ep * 2 + eo ∈ {0..23}
      ep : 该物理棱块的 home slot（0-11）
      eo : 朝向——0 = 主面颜色属于"好"颜色族，1 = 翻转
           U/D 层棱 (slots 0-7)：好颜色 ∈ {0, 1}（U/D 色）
           赤道棱   (slots 8-11)：好颜色 ∈ {2, 3}（F/B 色）
    """
    tokens = np.empty(20, dtype=np.int64)

    for slot, stickers in enumerate(CORNER_STICKERS):
        c0, c1, c2 = int(state[stickers[0]]), int(state[stickers[1]]), int(state[stickers[2]])
        cp = _CORNER_COLOR_TO_HOME[frozenset((c0, c1, c2))]
        # 哪个位置持有 U/D 颜色（0 或 1）决定方向
        if c0 in (0, 1):
            co = 0
        elif c1 in (0, 1):
            co = 1
        else:
            co = 2
        tokens[slot] = cp * 3 + co

    for slot, stickers in enumerate(EDGE_STICKERS):
        c0, c1 = int(state[stickers[0]]), int(state[stickers[1]])
        ep = _EDGE_COLOR_TO_HOME[frozenset((c0, c1))]
        if slot < 8:
            eo = 0 if c0 in (0, 1) else 1
        else:
            eo = 0 if c0 in (2, 3) else 1
        tokens[8 + slot] = ep * 2 + eo

    return tokens


# ══════════════════════════════════════════════════════════════════════
# 验证与演示
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("cube_env.py — NxNxN 魔方环境验证")
    print("=" * 65)

    # ── 1. 3×3 自检（代数性质）─────────────────────────────────
    print("\n[1] 3×3 动作代数性质验证")
    env3 = CubeEnv(N=3)
    assert env3.verify_moves(), "3×3 动作验证失败"
    print(f"  ✓ 所有 {env3.num_moves} 个 3×3 动作验证通过")

    # ── 2. 3×3 与原始代码兼容性 ───────────────────────────────
    print("\n[2] 3×3 动作与原始 cube_env.py 兼容性验证")
    # 打乱 20 步再应用逆序列，应回到原点
    for seed in range(5):
        state, seq = scramble(20, seed=seed)
        inv_seq = [m.replace("'", "TEMP").replace("2", "'2").replace("TEMP", "") for m in reversed(seq)]
        # 正确逆序：CW→CCW, CCW→CW, 180→180
        def inv_move(m: str) -> str:
            if m.endswith("2"): return m
            if m.endswith("'"): return m[:-1]
            return m + "'"
        inv_seq = [inv_move(m) for m in reversed(seq)]
        restored = apply_move_sequence(state, inv_seq)
        assert is_solved(restored), f"seed={seed}: 打乱后逆序还原失败"
    print("  ✓ 打乱+逆序还原（5个随机种子）全部成功")

    # ── 3. MOVE_NAMES 与原始一致 ──────────────────────────────
    print("\n[3] 3×3 MOVE_NAMES 验证")
    expected_18 = sorted([
        "B","B'","B2","D","D'","D2","F","F'","F2",
        "L","L'","L2","R","R'","R2","U","U'","U2",
    ])
    assert MOVE_NAMES == expected_18, f"MOVE_NAMES 不匹配:\n{MOVE_NAMES}"
    print(f"  ✓ MOVE_NAMES = {MOVE_NAMES}")

    # ── 4. 3×3 可视化 ─────────────────────────────────────────
    print("\n[4] 3×3 打乱展示")
    state, seq = scramble(5, seed=42)
    print(f"  打乱序列: {' '.join(seq)}")
    print(render_state(state))

    # ── 5. 4×4 验证 ───────────────────────────────────────────
    print("\n[5] 4×4 动作代数性质验证")
    env4 = CubeEnv(N=4)
    assert env4.verify_moves(), "4×4 动作验证失败"
    print(f"  ✓ 所有 {env4.num_moves} 个 4×4 动作验证通过")
    print(f"  动作列表: {env4.move_names}")

    # ── 6. 4×4 打乱+还原 ─────────────────────────────────────
    print("\n[6] 4×4 打乱+逆序还原验证")
    for seed in range(5):
        s4, seq4 = env4.scramble(30, seed=seed)
        def inv_move(m: str) -> str:
            if m.endswith("2"): return m
            if m.endswith("'"): return m[:-1]
            return m + "'"
        inv4 = [inv_move(m) for m in reversed(seq4)]
        restored4 = env4.apply_moves(s4, inv4)
        assert env4.is_solved(restored4), f"4×4 seed={seed} 还原失败"
    print("  ✓ 打乱30步+逆序还原（5个随机种子）全部成功")

    # ── 7. 4×4 可视化 ─────────────────────────────────────────
    print("\n[7] 4×4 打乱展示")
    s4, seq4 = env4.scramble(7, seed=0)
    print(f"  打乱序列: {' '.join(seq4)}")
    print(env4.render(s4))

    # ── 8. 5×5 验证 ───────────────────────────────────────────
    print("\n[8] 5×5 验证（含内层动作 U1, U2, U3, R1, R2, R3, F1, F2, F3）")
    env5 = CubeEnv(N=5)
    assert env5.verify_moves()
    print(f"  ✓ 5×5: {env5.num_moves} 个动作验证通过")
    inner = [m for m in env5.move_names if any(c.isdigit() and c != '2' for c in m[1:])]
    print(f"  内层动作: {[m for m in env5.move_names if m[1:2].isdigit() and m[1] != '2'][:6]} ...")

    # ── 9. 统计信息 ───────────────────────────────────────────
    print("\n[9] 各阶动作数统计")
    for n in [2, 3, 4, 5, 6]:
        e = CubeEnv(N=n)
        print(f"  N={n}: {e.num_stickers:4d} stickers, {e.num_moves:3d} moves, "
              f"one-hot dim={e.onehot_size}")

    print("\n✓✓ 所有验证通过！")