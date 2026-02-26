from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# __enter__, with yield, __exit__ pattern 去设置 torch 的默认 dtype，确保在 with 块内使用指定的 dtype，块结束后恢复原来的 dtype
# 用于切换精度
@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)



# 这是一个用于 NVIDIA NVTX（NVIDIA Tools Extension）性能分析标注的装饰器工厂函数。
# 用于注释forward函数，方便在性能分析工具中识别不同层的执行时间和调用关系。
def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
