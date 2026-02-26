from __future__ import annotations

import re
from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch


# 类型别名，表示一个 {参数名: 张量} 的字典，用于存储模型权重。
_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name

# BaseOP
#   --state_dict() 导出权重
#   --load_state_dict()-加载权重(支持MoE experts特殊处理)
# StateLessOP(BaseoP) 无参数层
# OPList(BaseOP) 层的列表容器
# 轻量级的PyTorch 的 nn.Module去除了训练时的参数只能用于推理
# BaseOP 定义了一个接口，要求子类实现 forward 方法，并提供了 state_dict 和 load_state_dict 方法来管理权重。StateLessOP 是一个特殊的 BaseOP，没有可训练参数，因此重写了 state_dict 和 load_state_dict 方法以适应这种情况。OPList 则是一个容器同nn.ModuleList，可以包含多个 BaseOP 实例，并通过索引管理它们的权重。
class BaseOP:
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue

            if isinstance(param, torch.Tensor):
                if "experts" in prefix:
                    mapped_name = name
                    matched_keys = []
                    for key in list(state_dict.keys()):
                        if prefix in key and mapped_name in key:
                            matched_keys.append(key)

                    def extract_expert_index(k):
                        match = re.search(r"experts\.(\d+)\.", k)
                        return int(match.group(1)) if match else 0

                    matched_keys.sort(key=extract_expert_index)

                    items = []
                    for k in matched_keys:
                        items.append(state_dict.pop(k))

                    if not items:
                        raise ValueError(
                            f"No weights found in state_dict for {prefix} and {mapped_name}"
                        )

                    item = torch.stack(items, dim=0)
                else:
                    item = state_dict.pop(_concat_prefix(prefix, name))

                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape and param.dtype == item.dtype

                setattr(self, name, item)

            elif isinstance(param, BaseOP):
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

# 像 ReLU、Dropout 这类不需要训练参数的层，重写方法返回空字典
class StateLessOP(BaseOP):
    def __init__(self):
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not _internal and state_dict:
            _ = prefix
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        _ = prefix
        return result if result is not None else {}


T = TypeVar("T", bound=BaseOP)


# 类似 PyTorch 的 nn.ModuleList，用索引作为前缀管理多个子层
class OPList(BaseOP, Generic[T]):
    def __init__(self, ops: List[T]):
        super().__init__()
        self.op_list = ops

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
