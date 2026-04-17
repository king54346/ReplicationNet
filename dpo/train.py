import os
import sys

from .dataset import DPODataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")


# ==========工具函数==========

def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def Logger(msg):
    if is_main_process():
        print(msg, flush=True)


def setup_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed_mode():
    local_rank = 0
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        Logger(f"分布式初始化完成，local_rank={local_rank}, world_size={dist.get_world_size()}")
    return local_rank


def get_lr(current_step, total_steps, base_lr, min_lr_ratio=0.1):
    """余弦退火学习率，从 base_lr 衰减到 base_lr * min_lr_ratio"""
    import math
    min_lr = base_lr * min_lr_ratio
    if current_step >= total_steps:
        return min_lr
    progress = current_step / total_steps
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine_decay


class SkipBatchSampler:
    """跳过前 skip_batches 个 batch，用于断点续训"""
    def __init__(self, sampler, batch_size, skip_batches):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch, skipped = [], 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        return max(0, len(self.sampler) // self.batch_size - self.skip_batches)


def save_checkpoint(
    weight,
    model=None,
    optimizer=None,
    scaler=None,
    epoch=None,
    step=None,
    wandb=None,
    save_dir="./checkpoints",
):
    """
    保存或加载 checkpoint。
    model=None 时为读取模式，返回 dict 或 None。
    """
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = os.path.join(save_dir, f"{weight}_ckp.pt")

    if model is None:
        if os.path.exists(ckp_path):
            Logger(f"加载 checkpoint: {ckp_path}")
            return torch.load(ckp_path, map_location="cpu")
        Logger(f"未找到 checkpoint: {ckp_path}，从头开始")
        return None

    model_state = (
        model.module.state_dict()
        if isinstance(model, DistributedDataParallel)
        else model.state_dict()
    )
    ckp = {
        "model":     model_state,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scaler":    scaler.state_dict() if scaler else None,
        "epoch":     epoch,
        "step":      step,
        "wandb_id":  wandb.id if wandb is not None else None,
    }
    torch.save(ckp, ckp_path)
    Logger(f"Checkpoint 已保存: {ckp_path}")


# ==========DPO 核心函数==========
# 具体例子
# 假设：
#
# Prompt x = "1+1等于"
# 回答 y = "等 于 2" （3个token）
#
# 模型对每个 token 给出概率：
# 步骤  预测      概率       log prob
# t=1p ("等" | x) 0.6       -0.51
# t=2p("于" | x, "等")   0.9  -0.11
# t=3p("2" | x, "等于")   0.8     -0.22
# π(y) = (-0.51) + (-0.11) + (-0.22) = -0.84
# LLM 最后一层输出的是 logits 1. logits → prob（Softmax）   2. prob → log prob 3. 取出真实 token 对应的 log prob  log_prob[234] = -0.51
#  为什么用log prob呢： 1. 防止梯度过小 2， 概率相乘对导致数值下溢
def logits_to_log_probs(logits, labels):
    """
    从 logits [B, S, V] 中取出 labels [B, S] 对应位置的 log 概率。
    返回 [B, S]。
    注意：logits 是预测"下一个 token"的概率，所以要错位对齐：
      logits[:, :-1, :] 对应 labels[:, 1:]（即用前一个位置预测当前）。
    这里直接输入的 labels 已经是对齐好的（Dataset 里 y 就是 input_ids 本身），
    所以直接 gather 即可。
    """
    log_probs = F.log_softmax(logits, dim=-1)           # [B, S, V]
    log_probs_per_token = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)
    ).squeeze(-1)                                        # [B, S]
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    DPO Loss：
    L = -log σ( β * ( (π(y_w) - π_ref(y_w)) - (π(y_l) - π_ref(y_l)) ) )
    y_w   preferred response（人类偏好的"好"回答）
    y_l   rejected response（人类不偏好的"差"回答）
    π(y)  当前训练中的 policy model 对回答 y 的 log probability
    π_ref(y) 冻结的reference model（通常是 SFT 模型）的 log probability
    β   控制偏离 reference model 程度的超参数
    σ   sigmoid 函数

    输入的 ref_log_probs / policy_log_probs 形状 [2B, S]，
    前半 B 是 chosen，后半 B 是 rejected。
    """
    # 序列长度（每条序列有效 token 数），防止除零
    seq_lengths = mask.sum(dim=1).clamp_min(1.0)           # [2B]

    # 对有效位置的 log 概率求均值，得到每条序列的标量 log 概率
    ref_logp    = (ref_log_probs    * mask).sum(dim=1) / seq_lengths   # [2B]
    policy_logp = (policy_log_probs * mask).sum(dim=1) / seq_lengths   # [2B]

    B = ref_logp.shape[0] // 2
    chosen_ref_logp    = ref_logp[:B]
    rejected_ref_logp  = ref_logp[B:]
    chosen_policy_logp = policy_logp[:B]
    rejected_policy_logp = policy_logp[B:]

    # 策略模型偏好差：chosen - rejected
    pi_logratios  = chosen_policy_logp  - rejected_policy_logp   # [B]
    # 参考模型偏好差：chosen - rejected（作为基准）
    ref_logratios = chosen_ref_logp     - rejected_ref_logp       # [B]

    # DPO 核心：策略相对于参考的"净偏好"
    logits = pi_logratios - ref_logratios                          # [B]
    loss   = -F.logsigmoid(beta * logits)                         # [B]
    return loss.mean()


# ==========训练一个 Epoch==========

def train_epoch(
    epoch,
    loader,
    iters,
    model,
    ref_model,
    optimizer,
    scaler,
    autocast_ctx,
    args,
    start_step=0,
    wandb=None,
):
    model.train()
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen   = batch["x_chosen"].to(args.device)
        x_rejected = batch["x_rejected"].to(args.device)
        y_chosen   = batch["y_chosen"].to(args.device)
        y_rejected = batch["y_rejected"].to(args.device)
        mask_chosen   = batch["mask_chosen"].to(args.device)
        mask_rejected = batch["mask_rejected"].to(args.device)
        attn_chosen   = batch["attention_mask_chosen"].to(args.device)
        attn_rejected = batch["attention_mask_rejected"].to(args.device)

        # chosen 和 rejected 拼成一个 batch，一次 forward 搞定
        x            = torch.cat([x_chosen,   x_rejected],   dim=0)   # [2B, S]
        y            = torch.cat([y_chosen,   y_rejected],   dim=0)   # [2B, S]
        mask         = torch.cat([mask_chosen, mask_rejected], dim=0)  # [2B, S]
        attention_mask = torch.cat([attn_chosen, attn_rejected], dim=0)  # [2B, S]

        # 余弦退火学习率（手动更新，保留原版风格）
        lr = get_lr(
            current_step=epoch * iters + step,
            total_steps=args.epochs * iters,
            base_lr=args.learning_rate,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with autocast_ctx:
            # ——— 参考模型前向（冻结，无梯度）———
            with torch.no_grad():
                ref_out    = ref_model(x, attention_mask=attention_mask)
                ref_logits = ref_out.logits                          # [2B, S, V]

            # ——— 策略模型前向 ———
            out    = model(x, attention_mask=attention_mask)
            logits = out.logits                                      # [2B, S, V]

            # 原版 DPODataset 已做自回归对齐：x=[:-1], y=[1:]
            # 所以这里 logits 和 y 维度已经匹配，直接 gather 即可
            ref_log_probs    = logits_to_log_probs(ref_logits, y)   # [2B, S]
            policy_log_probs = logits_to_log_probs(logits,     y)   # [2B, S]

            # DPO Loss（mask 与 y 已经对齐，直接传入）
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=args.beta)

            # MoE 辅助损失（非 MoE 模型 aux_loss=0）
            aux_loss = getattr(out, "aux_loss", torch.tensor(0.0, device=args.device))
            loss = (dpo_loss_val + aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ——— 日志 ———
        if is_main_process() and (step % args.log_interval == 0 or step == iters):
            elapsed   = time.time() - start_time
            cur_loss  = loss.item() * args.accumulation_steps
            cur_lr    = optimizer.param_groups[-1]["lr"]
            eta_min   = elapsed / (step + 1) * iters // 60 - elapsed // 60

            Logger(
                f"Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) "
                f"loss:{cur_loss:.6f} lr:{cur_lr:.2e} eta:{eta_min:.0f}min"
            )
            if wandb:
                wandb.log({"loss": cur_loss, "lr": cur_lr, "eta_min": eta_min})

        # ——— 保存 checkpoint ———
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            os.makedirs(args.save_dir, exist_ok=True)

            # 保存推理用权重（半精度）
            ckp_path = os.path.join(args.save_dir, f"dpo_qwen3_step{step}.pth")
            sd = (
                model.module.state_dict()
                if isinstance(model, DistributedDataParallel)
                else model.state_dict()
            )
            torch.save({k: v.half() for k, v in sd.items()}, ckp_path)
            Logger(f"模型权重已保存: {ckp_path}")

            # 保存完整训练状态（含优化器，用于续训）
            save_checkpoint(
                weight="dpo_qwen3",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=os.path.join(args.save_dir, "checkpoints"),
            )
            model.train()


# ==========主函数==========
# python dpo_qwen3.py \
#     --model_path /path/to/Qwen3-4B \
#     --data_path ./dataset/dpo.jsonl \
#     --save_dir ./out/dpo_qwen3 \
#     --epochs 1 \
#     --batch_size 1 \
#     --accumulation_steps 8 \
#     --max_seq_len 2048 \
#     --beta 0.1 \
#     --learning_rate 4e-8 \
#     --dtype bfloat16 \
#     --log_interval 10 \
#     --save_interval 100

# torchrun --nproc_per_node=2 dpo_qwen3.py \
#     --model_path /path/to/Qwen3-4B \
#     --data_path ./dataset/dpo.jsonl \
#     --save_dir ./out/dpo_qwen3 \
#     --batch_size 1 \
#     --accumulation_steps 4
if __name__ == "__main__":
    """
    DPO 训练主函数（基于 Qwen3-4B）

    DPO 训练流程：
    1. 加载策略模型（待优化）和参考模型（冻结的 SFT 基础模型）
    2. 加载偏好数据（chosen 好回答 vs rejected 差回答）
    3. 同时对两种回答做 forward，计算 DPO loss
    4. 只更新策略模型，参考模型始终冻结
    5. DPO 不需要 Reward 模型，直接端到端优化偏好
    """
    parser = argparse.ArgumentParser(description="Qwen3-4B DPO 训练")

    # ===== 模型路径 =====
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B",
                        help="Qwen3 模型路径（HuggingFace Hub ID 或本地路径）")

    # ===== 数据 =====
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl",
                        help="DPO 数据路径，每行格式：{chosen: [...], rejected: [...]}")

    # ===== 保存 =====
    parser.add_argument("--save_dir", type=str, default="../out/dpo_qwen3")

    # ===== 训练基础参数 =====
    parser.add_argument("--epochs", type=int, default=1,
                        help="训练轮数（DPO 通常 1-2 轮）")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="每个 batch 的样本数（实际显存占用是 2x，因为 chosen+rejected 拼在一起）")
    parser.add_argument("--learning_rate", type=float, default=4e-8,
                        help="初始学习率（DPO 建议 ≤ 5e-8，过大容易遗忘）")
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_seq_len", type=int, default=4096)

    # ===== DPO 超参数 =====
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta：控制偏好优化强度。0.1-0.5 常用，越大越激进")

    # ===== 硬件 =====
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--num_workers", type=int, default=1)

    # ===== 日志与续训 =====
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Qwen3-DPO")

    args = parser.parse_args()

    # ========== 1. 初始化分布式 & 随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device_type = "cuda" if "cuda" in args.device else "cpu"
    autocast_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ========== 2. 检查 checkpoint ==========
    ckp_data = None
    if args.from_resume == 1:
        ckp_data = save_checkpoint(
            weight="dpo_qwen3",
            save_dir=os.path.join(args.save_dir, "checkpoints"),
        )

    # ========== 3. 初始化 wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        wandb.init(
            project=args.wandb_project,
            name=f"qwen3-dpo-bs{args.batch_size}-lr{args.learning_rate}-beta{args.beta}",
            id=wandb_id,
            resume="must" if wandb_id else None,
        )

    # ========== 4. 加载 Tokenizer ==========
    Logger(f"加载 Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        Logger("pad_token 未设置，已自动设为 eos_token")

    # ========== 5. 加载策略模型（Policy，待优化）==========
    Logger(f"加载策略模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=args.device,
    )
    Logger(f"策略模型参数量：{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # ========== 6. 加载参考模型（Reference，冻结）==========
    # 参考模型和策略模型初始权重完全相同，但训练过程中始终冻结
    # 它代表"优化前的基准"，用于防止策略模型在优化偏好时过度偏离原始分布
    Logger("加载参考模型（冻结）...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=args.device,
    ).eval().requires_grad_(False)
    Logger(f"参考模型参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e9:.2f}B")

    # ========== 7. 数据集 & 优化器 ==========
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # GradScaler 只在 float16 下启用；bfloat16 不需要（数值范围够大）
    scaler    = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # ========== 8. 从 checkpoint 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        if ckp_data.get("scaler") and args.dtype == "float16":
            scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step  = ckp_data.get("step", 0)
        Logger(f"从 checkpoint 恢复：epoch={start_epoch}, step={start_step}")

    # ========== 9. DDP 包装 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
        ref_model.to(args.device)

    # 计算总 iters（用一个临时 loader 统计）
    _tmp_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(_tmp_loader)
    Logger(f"训练开始：epochs={args.epochs}, iters/epoch={iters}")

    # ========== 10. 训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)),
                args.batch_size,
                start_step,
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(f"Epoch {epoch+1}: 跳过前 {start_step} 步，从 step {start_step+1} 继续")
            train_epoch(
                epoch, loader, len(loader) + start_step,
                model, ref_model, optimizer, scaler, autocast_ctx,
                args, start_step, wandb,
            )
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            train_epoch(
                epoch, loader, iters,
                model, ref_model, optimizer, scaler, autocast_ctx,
                args, 0, wandb,
            )

    Logger("DPO 训练完成！")