from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)

# 激活+乘法的封装函数， out参数是可选的，如果提供了out参数，函数会将结果直接写入out中，避免了额外的内存分配和数据复制，从而提升性能。
# 通常用于Gated FFN
def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
