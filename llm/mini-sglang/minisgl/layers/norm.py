from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)
    # 原地修改 x省显存，推理时常用
    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)

# 正常写法需要两步： 先加上残差，再做 RMSNorm(x = x + residual, x = rmsnorm(x) )；而 fused_add_rmsnorm 则将两步融合成一步，减少了内存访问和计算开销，从而提升性能。
# 第一次没有任何残差，residual=None，直接进行 RMSNorm 计算和原始x的返回；后续层有残差，先进行残差加法，再进行 RMSNorm 计算，最后返回归一化结果和残差加法后的结果。
class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if residual is None:
            # 没有残差, 直接进行 RMSNorm 计算和原始x的返回
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        # x 是归一化结果，residual 是残差加法后的结果
        return x, residual
