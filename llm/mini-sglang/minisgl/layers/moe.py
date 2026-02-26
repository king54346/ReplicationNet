from typing import Optional

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int, # 专家总数
        top_k: int,  # 每个 token 激活几个专家
        hidden_size: int,   # 输入/输出维度
        intermediate_size: int, # 每个专家内部 FFN 的中间维度
        layer_id: Optional[int] = None,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.params_dtype = params_dtype
        self._comm = DistributedCommunicator()
        # 张量并行，把每个专家的 intermediate_size 均分到各卡上，减少单卡显存压力。
        tp_info = get_tp_info()
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.layer_id = layer_id
        intermediate_size_per_partition = div_even(intermediate_size, tp_size)
        self.gate_up_proj = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype,
        )
        self.down_proj = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype,
        )
    # Router Logits是路由器打分 moe_backend 会做 softmax + top-k 选出每个 token 对应的专家
    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        ctx = get_global_ctx()
        # moe_backend 负责专家调度和计算
        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
        if self.tp_size > 1:
            # 多卡并行时，各卡只算了部分 intermediate，最后需要 `all_reduce` 汇总
            #                     输入 x [hidden_size]
            #                    /                    \
            #               GPU 0                    GPU 1
            #      gate_up_proj[:, :inter/2, :]   gate_up_proj[:, inter/2:, :]
            #               ↓                            ↓
            #          算前半段                       算后半段
            #          partial result 0             partial result 1
            #                    \                    /
            #                     all_reduce (求和)
            #                           ↓
            #                    完整的 output [hidden_size]
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        return final_hidden_states
