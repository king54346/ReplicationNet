"""
基于 TRL DPOTrainer 的 Qwen3 DPO 训练脚本

单卡训练：
  python train.py --model_path Qwen/Qwen3-4B --data_path ./dpo.jsonl

多卡训练（torchrun / accelerate）：
  accelerate launch train.py --model_path Qwen/Qwen3-4B --data_path ./dpo.jsonl

LoRA 模式（显存不足时）：
  python train.py --model_path Qwen/Qwen3-4B --use_lora
"""

import os
import sys
import argparse
import warnings

import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ──────────────────────────────────────────────────────────
# 数据预处理：把原始格式转为 TRL 要求的 prompt/chosen/rejected
# ──────────────────────────────────────────────────────────
# 支持两种输入格式，自动识别：
#
# 格式 A（btfChinese_DPO.jsonl，扁平字符串）：
#   {"system": "", "question": "...", "chosen": "...", "rejected": "..."}
#
# 格式 B（dpo.jsonl，完整对话列表）：
#   {"chosen": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}],
#    "rejected": [...]}
#
# 统一输出为 TRL 格式：
#   {"prompt":   [{"role": "user", "content": "..."}],
#    "chosen":   [{"role": "assistant", "content": "..."}],
#    "rejected": [{"role": "assistant", "content": "..."}]}
# ──────────────────────────────────────────────────────────
def preprocess(example: dict) -> dict | None:
    chosen  = example["chosen"]
    rejected = example["rejected"]

    # ── 格式 A：chosen/rejected 是字符串 ──
    if isinstance(chosen, str):
        prompt = []
        if example.get("system", ""):
            prompt.append({"role": "system", "content": example["system"]})
        if example.get("question", ""):
            prompt.append({"role": "user", "content": example["question"]})
        if not prompt:
            return None
        return {
            "prompt":   prompt,
            "chosen":   [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        }

    # ── 格式 B：chosen/rejected 是完整对话列表 ──
    last_assistant_idx = max(
        (i for i, m in enumerate(chosen) if m["role"] == "assistant"), default=-1
    )
    if last_assistant_idx == -1:
        return None
    rej_assistants = [m for m in rejected if m["role"] == "assistant"]
    if not rej_assistants:
        return None
    return {
        "prompt":   chosen[:last_assistant_idx],
        "chosen":   [chosen[last_assistant_idx]],
        "rejected": [rej_assistants[-1]],
    }


def build_dataset(data_path: str) -> Dataset:
    """
    用标准 json 逐行读取 JSONL，绕开 datasets schema 推断导致的
    Sequence(Struct) → dict-of-lists 类型转换问题。
    """
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)   # list[dict] 结构完全可控
            result = preprocess(example)
            if result is not None:
                records.append(result)

    ds = Dataset.from_list(records)
    print(f"数据集加载完成，共 {len(ds)} 条样本")
    print("示例：", ds[0])
    return ds


# ──────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3 DPO 训练（基于 TRL）")

    # ── 模型与数据 ──
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--data_path",  type=str, default="./btfChinese_DPO.jsonl")
    parser.add_argument("--save_dir",   type=str, default="../out/dpo_qwen3")

    # ── SFT 预热（先用 chosen 做有监督微调，打破基础模型的 RLHF 对齐）──
    parser.add_argument("--sft_first",      action="store_true",
                        help="DPO 训练前先用 chosen 回答做一轮 SFT，帮助突破基础模型对齐")
    parser.add_argument("--sft_epochs",     type=int,   default=1)
    parser.add_argument("--sft_lr",         type=float, default=2e-5)
    parser.add_argument("--sft_save_dir",   type=str,   default="../out/sft_qwen3")

    # ── DPO 超参 ──
    parser.add_argument("--beta",            type=float, default=0.05,
                        help="DPO beta：越小偏离参考模型越激进（常用 0.01~0.1）")
    parser.add_argument("--max_length",      type=int,   default=2048,
                        help="chosen/rejected 最大总长度（prompt + response）")

    # ── 训练基础参数 ──
    parser.add_argument("--epochs",           type=int,   default=1)
    parser.add_argument("--batch_size",       type=int,   default=1,
                        help="per_device_train_batch_size")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate",    type=float, default=5e-6)
    parser.add_argument("--grad_clip",        type=float, default=1.0)
    parser.add_argument("--warmup_ratio",     type=float, default=0.1,
                        help="warmup 占总步数的比例（替代固定步数，自动适配数据集大小）")

    # ── 硬件 ──
    parser.add_argument("--dtype",       type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_workers", type=int, default=4)

    # ── LoRA（显存不足时开启）──
    parser.add_argument("--use_lora",    action="store_true")
    parser.add_argument("--lora_r",      type=int,   default=16)
    parser.add_argument("--lora_alpha",  type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # ── 日志与续训 ──
    parser.add_argument("--log_interval",  type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--from_resume",   action="store_true",
                        help="从 save_dir 下最新 checkpoint 续训")
    parser.add_argument("--use_wandb",     action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Qwen3-DPO")

    args = parser.parse_args()

    # ── 1. 数据集 ──
    dataset = build_dataset(args.data_path)

    # ── 2. Tokenizer ──
    print(f"加载 Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 3. 模型 ──
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # accelerate launch 时每个进程有自己的 LOCAL_RANK，模型加载到对应 GPU
    # 不能用 device_map="auto"，否则 Trainer 的 model.to(device) 会 OOM
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank} if torch.cuda.is_available() else None

    print(f"加载策略模型: {args.model_path}  (local_rank={local_rank})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # ── 4. SFT 预热（可选）──
    # 用 chosen 回答做有监督微调，让模型先学会生成 chosen 风格的输出，
    # 再用 DPO 强化偏好，效果远好于直接 DPO 对抗强 RLHF 对齐。
    if args.sft_first:
        print("\n===== 阶段 1/2：SFT 预热 =====")

        # SFT 数据：把 prompt + chosen 拼成完整对话
        def make_sft_messages(example):
            return {"messages": example["prompt"] + example["chosen"]}

        sft_ds = dataset.map(make_sft_messages, remove_columns=dataset.column_names)

        sft_args = SFTConfig(
            output_dir=args.sft_save_dir,
            num_train_epochs=args.sft_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.accumulation_steps,
            learning_rate=args.sft_lr,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",
            bf16=(args.dtype == "bfloat16"),
            fp16=(args.dtype == "float16"),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False,
            logging_steps=args.log_interval,
            save_steps=args.save_interval,
            save_total_limit=2,
            report_to="wandb" if args.use_wandb else "none",
        )
        sft_trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=sft_ds,
            processing_class=tokenizer,
        )
        sft_trainer.train()
        sft_trainer.save_model(args.sft_save_dir)
        print(f"SFT 模型已保存至: {args.sft_save_dir}")
        print("===== 阶段 2/2：DPO 训练 =====\n")

    # ── 5. LoRA（可选）──
    peft_config = None
    if args.use_lora:
        from peft import LoraConfig, TaskType
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        print(f"LoRA 已启用: r={args.lora_r}, alpha={args.lora_alpha}")

    # ── 6. DPOConfig ──
    # 全参微调时不显式传 ref_model，改用 precompute_ref_log_probs=True：
    #   TRL 在训练前用当前模型初始权重把所有 ref log probs 算好缓存到 dataset，
    #   然后把 ref model 从 GPU 完全卸载，训练阶段只占一份模型显存。
    resume_from = args.save_dir if args.from_resume else None

    training_args = DPOConfig(
        # 输出
        output_dir=args.save_dir,

        # DPO 核心
        beta=args.beta,
        max_length=args.max_length,
        loss_type="sigmoid",

        # 显存优化：预计算 ref log probs，训练时不再持有 ref model
        precompute_ref_log_probs=True,

        # 训练基础参数
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",

        # 精度与内存
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # DDP 下必须关闭 reentrant
        ddp_find_unused_parameters=False,   # 减少 DDP 通信桶的额外内存开销
        dataloader_num_workers=args.num_workers,

        # 日志与保存
        logging_steps=args.log_interval,
        save_steps=args.save_interval,
        save_total_limit=3,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"qwen3-dpo-lr{args.learning_rate}-beta{args.beta}",

        # 续训
        resume_from_checkpoint=resume_from,
    )

    # ── 6. DPOTrainer ──
    # ref_model=None：TRL 用策略模型的初始权重作为参考，配合 precompute_ref_log_probs 使用
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # ── 8. 训练 ──
    print("开始 DPO 训练...")
    trainer.train(resume_from_checkpoint=resume_from)

    # ── 9. 保存 ──
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"模型已保存至: {args.save_dir}")

    # ── 10. 清理分布式进程组（消除 NCCL 资源泄漏 warning）──
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
