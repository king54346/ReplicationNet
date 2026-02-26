from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP

# 张量并行下的各种线性层切分实现
# Attention 里：
#   QKV 投影  → LinearQKVMerged    （列并行，输出切）
#   O  投影   → LinearOProj        （行并行，输入切，all_reduce）
#
# FFN 里：
#   gate+up  → LinearColParallelMerged （列并行，输出切）
#   down     → LinearRowParallel        （行并行，输入切，all_reduce）
#
# Router 里：
#   打分     → LinearReplicated         （不切，每卡完整副本）
# 行切分和列切分的区别：
# 列并行 (只需要切W)
#          col0 col1 col2 col3 col4 col5
# row0  [  1    2    3    4    5    6  ]
# row1  [  7    8    9   10   11   12  ]
# row2  [ 13   14   15   16   17   18  ]
# row3  [ 19   20   21   22   23   24  ]
#
# GPU0 拿左半：W_left  [4, 3]    GPU1 拿右半：W_right [4, 3]
#          col0 col1 col2                  col3 col4 col5
# row0  [  1    2    3  ]         row0  [  4    5    6  ]
# row1  [  7    8    9  ]         row1  [ 10   11   12  ]
# row2  [ 13   14   15  ]         row2  [ 16   17   18  ]
# row3  [ 19   20   21  ]         row3  [ 22   23   24  ]
# GPU0: x @ W_left  = [1*1+2*7+3*13+4*19, ...] = [out0, out1, out2]
# GPU1: x @ W_right = [1*4+2*10+3*16+4*22, ...]= [out3, out4, out5]
# [out0, out1, out2] + [out3, out4, out5] = [out0, out1, out2, out3, out4, out5]
# 行切分 (注意: W和x都要切)：
# 完整 W [4, 6]：
#
# GPU0 拿上半：W_top [2, 6]        GPU1 拿下半：W_bot [2, 6]
#          col0~col5                        col0~col5
# row0  [  1  2  3  4  5  6 ]     row2  [ 13 14 15 16 17 18 ]
# row1  [  7  8  9 10 11 12 ]     row3  [ 19 20 21 22 23 24 ]
#
# x 也跟着切：
# GPU0 拿: x_top = [1, 2]
# GPU1 拿: x_bot = [3, 4]
# GPU0: x_top @ W_top = [1*1+2*7,  1*2+2*8,  ...] = [GPU0_partial_0~5]
# GPU1: x_bot @ W_bot = [3*13+4*19, 3*14+4*20, ...] = [GPU1_partial_0~5]
# [GPU0_partial0+GPU1_partial0 , GPU0_partial1+GPU1_partial1, ...] = [out0, out1, out2, out3, out4, out5]


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

# 每卡完整副本
class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )

# 按列切分，多个输出合并
# 各卡输出的是不同的列 → 拼接(concat)即可，不用通信
# 行并行必须 all_reduce 相加才是完整结果
class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


# 专门为 GQA（Grouped Query Attention） 设计
# 每卡负责不同的头，比如 Q有32个head，KV有8个head，ratio=4
#   Q: 8 heads
#   K: 2 heads
#   V: 2 heads
class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        GQA_ratio = div_even(num_qo_heads, num_kv_heads)
        local_num_kv = div_even(num_kv_heads, tp_info.size)
        full_isize = hidden_size
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

# Attention 输出投影，行并行，专门给 Attention 的 O 矩阵用
# GPU0: 拿到前半段 attention 输出 → 算部分结果
# GPU1: 拿到后半段 attention 输出 → 算部分结果
#          ↓
#       all_reduce 求和 → 完整输出
class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y

# 通用行并行,用于FFN 里的 down_proj
class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
