from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP

# 旋转位置编码实现,让距离近的 token 点积更大，距离远的点积更小
class RotaryEmbedding(StateLessOP):
    #  预计算 cos/sin 缓存
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        #每个维度对应一个不同的旋转频率，低维度旋转快(高频)，高维度旋转慢(低频),(远距离靠高维度区分,近距离靠低维度区分)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float) # [0, 1, 2, ..., max_pos]
        freqs = torch.einsum("i,j -> ij", t, inv_freq) # 每个位置 × 每个频率
        cos = freqs.cos() # [max_pos, rotary_dim/2]
        sin = freqs.sin() # [max_pos, rotary_dim/2]
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)  # 拼起来备用
        assert self.head_size in [64, 128, 256, 512]

        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace
    #  原地旋转 Q 和 K
    #  两两配对运算[q0, q1] → [q0*cos - q1*sin, q0*sin + q1*cos]
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        return query, key

# 处理不同的 RoPE 变体
# `default`：标准 RoPE
#  `llama3`：LLaMA3 的长文本扩展
# 高频维度（旋转快）：  位置变化敏感 → 不缩放，保持原样
# 低频维度（旋转慢）：  适合编码长距离 → 除以 scaling_factor 拉伸
# 中间维度：           平滑过渡，不硬切
# `linear`：线性缩放
def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    
    # 处理不同的 rope_type
    rope_type = rope_scaling.get("rope_type", "default")
    
    match rope_type:
        case "default":
            # default 类型:使用 rope_theta 作为 base(如果提供)
            theta = rope_scaling.get("rope_theta", base)
            return RotaryEmbedding(head_dim, rotary_dim, max_position, theta)
        
        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            # llama3 可能也有自定义的 rope_theta
            theta = rope_scaling.get("rope_theta", base)
            return RotaryEmbedding(head_dim, rotary_dim, max_position, theta, post_process)
        
        case "linear" | "dynamic":
            # 其他常见类型的处理
            scaling_factor = rope_scaling.get("factor", 1.0)
            theta = rope_scaling.get("rope_theta", base)
            # linear/dynamic scaling 通常需要调整 max_position
            adjusted_max_pos = int(max_position * scaling_factor)
            return RotaryEmbedding(head_dim, rotary_dim, adjusted_max_pos, theta)
        
        case _:
            raise ValueError(f"Unsupported rope_type: {rope_type}, full config: {rope_scaling}")

_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
