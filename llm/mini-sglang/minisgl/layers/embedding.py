from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_ceil, nvtx_annotate

from .base import BaseOP

#  并行词嵌入
# 每个 GPU 负责 vocal/n 个 token 的 embedding
class VocabParallelEmbedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel import indexing
        # 在本GPU的词表范围内查找，不在范围内的位置补0
        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )
        # 所有GPU的结果求和（因为只有一个GPU有非零值，等效于gather）
        return self._comm.all_reduce(y) if self.tp_size > 1 else y

# 并行语言模型头：把hidden_states 投影回词表大小，是 Embedding 的逆操作（矩阵乘法）
class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,  #共享同一套权重 Embedding 是"词→向量"，LM Head 是"向量→词"
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        #  prefill 优化：prefill 阶段只需要最后一个 token 的 logits 来预测下一个词，不需要对所有输入 token 都做投影，**大幅节省计算量**
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        module = self.tied_embedding or self
        # 输出单GPU词表 logits向量
        logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        # 多卡并行时，每个 GPU 只算本GPU词表部分的 logits，最后需要 `all_gather` 汇总
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        if bs == 1:
            # 只有1个batch
            # 因为 `num_embeddings` 可能不是 `tp_size` 的整数倍（用了 `div_ceil` 做分块），最后一段可能有 padding，所以在这里裁掉多出来的部分
            return output_tensor.view(1, -1)[:, : self.num_embeddings]
        # 多于1个batch，先把 `tp_size` 个分块的结果拼成一个完整的词表维度，再裁掉多出来的部分
        # (tp_size, N, local_vocab)--->(N, local_vocab, tp_size)--->(N, tp_size * local_vocab)--->(N, num_embeddings)
        tp = self.tp_size
        n = input_shape[0]
        local_vocab = input_shape[1]
        # (tp, N, local_vocab) -> (N, tp * local_vocab)
        output_tensor = output_tensor.transpose(0, 1).reshape(n, tp * local_vocab)
        return output_tensor[:, : self.num_embeddings]
