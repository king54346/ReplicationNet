from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
import torch.nn.functional as F
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger
from transformers import AutoTokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput

logger = init_logger(__name__)


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    load_indices: torch.Tensor
    write_indices: torch.Tensor


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


# Scheduler 负责把请求从队列取出、组批、分配 KV 缓存、调度 prefill/ decode、调用模型推理，然后把结果回传给 detokenizer
# SchedulerIOMixin 负责收/发消息（和其他进程通信）
# 创建 Engine、初始化 ZMQ I/O、创建双 CUDA Stream（主推理 stream + engine_stream_ctx）、初始化 4 个子管理器（Table/Cache/Decode/Prefill）
class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine
        # 模型前向的执行器
        self.engine = Engine(config)
        # 初始化 SchedulerIOMixin，传入 config 和 self.engine.tp_cpu_group（一个 torch.distributed.ProcessGroup，用于在 CPU 上同步所有 TP rank）
        super().__init__(config, self.engine.tp_cpu_group)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # 管理 KV cache 的页表、内存分配与回收
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(self.device, self.engine.num_pages, config.cache_type)
        # 分别管理 prefill（输入阶段）和 decode（生成阶段）的请求队列与调度
        self.decode_manager = DecodeManager()
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        self.tp_info = config.tp_info
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.page_table = self.engine.page_table
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self.dummy_write_2d_pos = (self.engine.dummy_req.table_idx, 1, 2)  # 0 for load, 1 for write

    # 把上一轮生成的 token 包成 DetokenizeMsg 发送给 detokenizer，同时回收已经结束的请求缓存
    def _process_last_data(
            self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
        if last_data is None:
            return
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []

        for i, req in enumerate(batch.reqs):
            if req in self.finished_reqs or isinstance(req, ChunkedReq):
                continue

            next_token_id = next_tokens_cpu[i]
            req.append_host(next_token_id.unsqueeze(0))
            next_token = int(next_token_id.item())
            finished = not req.can_decode()
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id
            reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            # free resources if the req is finished and not ongoing
            if finished:
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)
                logger.debug_rank0("Request %s is finished", req)

        # free resources for finished but not ongoing reqs
        ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
        for req in self.finished_reqs.difference(ongoing_reqs):
            self.table_manager.free(req.table_idx)
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.input_ids[: req.cached_len],
                self.page_table[req.table_idx, : req.cached_len],
            )

        # keep only ongoing reqs in the finished set
        self.finished_reqs.intersection_update(ongoing_reqs)
        self.send_result(reply)

    # 检查长度并加入 prefill 队列
    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    # 为调度好的批次准备全部数据：分配 KV 页面 → CUDA Graph padding → 计算 2D→1D 索引 → 写页表 → 准备 attention 元数据和采样参数
    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        needed_size = sum(r.extend_len for r in batch.reqs)
        batch.out_loc = self.cache_manager.allocate(needed_size)
        # NOTE: Pad the batch if needed
        if padding_size := self.engine.graph_runner.pad_batch(batch):
            batch.out_loc = F.pad(batch.out_loc, (0, padding_size), value=self.engine.dummy_page)
        # NOTE: prepare 2d indices for token ids loading and writing
        load_indices = self._make_2d_indices(
            [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
        )
        write_indices = self._make_2d_indices(
            [
                (
                    (r.table_idx, r.device_len, r.device_len + 1)
                    if r.can_decode()  # NOTE: for chunked req, write to dummy pos
                    else self.dummy_write_2d_pos
                )
                for r in batch.reqs
            ]
        )
        assert all(r.device_len < self.engine.max_seq_len for r in batch.reqs)
        # NOTE: write out_loc to page_table before `prepare_metadata`
        self.page_table.view(-1)[load_indices] = batch.out_loc
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            load_indices=load_indices,
            write_indices=write_indices,
        )

    # 从 prefill 或 decode 队列选批次
    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
                self.prefill_manager.schedule_next_batch(self.prefill_budget)
                or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    # 将 token_pool 的 2D (行, 起始列, 结束列) 范围转换为 1D 扁平索引，用 pinned memory 异步传 GPU
    def _make_2d_indices(self, ranges: List[Tuple[int, int, int]]) -> torch.Tensor:
        """
        Return the 1D indices for the given 2D table and ranges.

        Example: The underlying indices of a 2D table (3, 4) are:
            [[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]]
        For ranges [(0, 1, 3), (2, 0, 2)], the returned indices are [1, 2, 8, 9].

        Args:
            ranges (List[Tuple[int, int, int]]): A list of tuples (entry, begin, end),
                where `entry` is the row index in the 2D table, and `begin` and `end`
                specify the range of column indices to include.
        Returns:
            torch.Tensor: A 1D tensor of indices.
        """
        STRIDE = self.token_pool.stride(0)
        needed_size = sum(end - begin for _, begin, end in ranges)
        indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
        offset = 0
        for entry, begin, end in ranges:
            length = end - begin
            offset += length
            torch.arange(
                begin + entry * STRIDE,
                end + entry * STRIDE,
                dtype=torch.int32,
                out=indices_host[offset - length: offset],
            )
        return indices_host.to(self.device, non_blocking=True)

    def _load_token_ids(self, input: ForwardInput) -> None:
        input.batch.input_ids = self.token_pool.view(-1)[input.load_indices]

    def _write_token_ids(self, input: ForwardInput, output: ForwardOutput) -> None:
        self.token_pool.view(-1)[input.write_indices] = output.next_tokens_gpu

    # 完成一次推理（加载 token ids → 前向 → 写回新 token）
    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        self._load_token_ids(forward_input)
        batch, sample_args = forward_input.batch, forward_input.sample_args
        if ENV.OVERLAP_EXTRA_SYNC:  # NOTE: https://github.com/sgl-project/mini-sglang/issues/58
            self.stream.synchronize()
        forward_output = self.engine.forward_batch(batch, sample_args)
        self._write_token_ids(forward_input, forward_output)
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    # 把上一批结果处理与当前批计算重叠，隐藏 CPU 延迟，提高 GPU 利用
    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        copy_done.synchronize()   # 同步的是 last_data 的拷贝完成事件

        prefill 阶段：GPU 跑得慢（长 prompt），CPU 处理结果反而很快
                    → overlap 收益大，CPU 早早处理完，GPU 还在跑

        decode 阶段：GPU 跑得快（1 token），CPU 处理结果也快
                    → overlap 收益相对小，但依然有效

        本轮的 ongoing_data 就是下一轮的 last_data
        last_data 是上一轮已完成的 batch，需要 copy_done.synchronize() 等数据拷到 CPU 后才能处理；
        ongoing_data 是本轮正在跑的 batch，CPU 不等它、不读它，只用它的请求列表来判断哪些 KV Cache 还不能释放。

        轮次  last_data  GPU 在跑              CPU 在做                     发送给Detokenizer
        ────────────────────────────────────────────────────────────────────────────────────
        0    None       prefill(req0,req1)    收消息+调度+提交              -
                        [异步启动]
        1    第0轮      (GPU 已完成prefill)   调度=None                    token42(req0)
                        (本轮无新提交)        等copy_done + 处理prefill结果  token87(req1)
        2    None       decode(req0,req1)     调度decode batch+提交         -
                        [异步启动]
        3    第2轮      (GPU 已完成decode)    等copy_done + 处理decode结果   token99(req0,fin)
                        (本轮无新提交)        释放req0,req1的KV Cache        token2 (req1,fin)
        """
        blocking = not (
                last_data  # don't block if we have a batch to be processed
                or self.prefill_manager.runnable
                or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data, ongoing_data)
        return ongoing_data

    # 单步处理消息 → 组批 → 前向 → 处理结果
    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data, None)

    # 决定用 normal loop 或 overlap loop
    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()
