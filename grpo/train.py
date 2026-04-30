"""
基于 TRL GRPOTrainer 的 Qwen3 对话强化 GRPO 训练脚本

数据格式（dataset2.py / rlaif.jsonl）：
  {"conversations": [
    {"role": "user",      "content": "基于以下角色信息完成一段对话..."},
    {"role": "assistant", "content": "张明：嗨，刘琳...（完整对话）"},
    {"role": "user",      "content": "基于以上对话提出一个问题。"},
    {"role": "assistant", "content": "这些智能家居产品需要哪些前提条件..."},
    {"role": "user",      "content": "请回答这个问题。"},
    {"role": "assistant", "content": ""}   ← 待生成，最后一条 assistant 为空
  ]}

特点：
  - 最后一条 assistant 内容为空，模型需根据上下文对话生成答案
  - 对话中第一条 assistant 长文本包含可用于软对比的背景知识
  - 倒数第二条 assistant 内容是本轮提出的问题（作为参考锚点）

单卡：
  python train.py --model_path ./model --data_path ./rlaif.jsonl

多卡：
  accelerate launch train.py --model_path ./model --data_path ./rlaif.jsonl
  reward 函数 选择 Skywork/Skywork-Reward-V2-Qwen3-1.7B 多了一个线性分类头
"""

import os
import sys
import argparse
import warnings

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset2 import RLAIFDataset


# ──────────────────────────────────────────────────────────
# 数据集构建
# ──────────────────────────────────────────────────────────
# 针对本数据格式的特殊处理：
#   - 最后一条 assistant 内容为空（待生成），不能用作参考答案
#   - 从对话历史中提取两类参考信息：
#       context_ref : 第一条 assistant 的长文本（背景对话，含事实答案）
#       question    : 倒数第二条 assistant 内容（本轮提出的问题）
#   - prompt = apply_chat_template(全部消息[:-1], add_generation_prompt=True)
# ──────────────────────────────────────────────────────────
def _extract_refs(conversations: list) -> tuple[str, str]:
    """
    从 conversations 列表中提取：
      context_ref : 首条 assistant 的长文本（包含背景事实）
      question    : 最后一条非空 assistant 内容（本轮提问）
    """
    context_ref = ""
    question = ""
    for turn in conversations:
        if turn.get("role") == "assistant" and turn.get("content", "").strip():
            if not context_ref:
                context_ref = turn["content"]   # 首条非空 assistant
            question = turn["content"]           # 持续更新→最后一条非空 assistant
    return context_ref, question

# 必填：prompt

# 支持两种格式：
#
# # 格式1：已渲染的字符串（dataset2 目前输出的就是这种）
# {"prompt": "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"}
#
# # 格式2：messages 列表（Trainer 内部会自动调用 apply_chat_template）
# {"prompt": [{"role": "user", "content": "你好"}]}
# 可选 reward_fn(context_ref=...) 收到


def build_dataset(data_path: str, tokenizer) -> Dataset:
    raw = RLAIFDataset(data_path, tokenizer)
    records = []
    for item in raw:
        prompt = item["prompt"]
        if not prompt:
            continue
        # 从原始样本中提取对话历史（dataset2 已加载 self.samples）
        sample = raw.samples[len(records)]  # 与迭代顺序一致
        conversations = sample.get("conversations", [])
        context_ref, question = _extract_refs(conversations)
        records.append({
            "prompt":      prompt,
            "context_ref": context_ref,   # 背景长文本，含事实答案
            "question":    question,      # 本轮提出的问题
        })

    ds = Dataset.from_list(records)
    print(f"数据集加载完成，共 {len(ds)} 条样本")
    return ds


# ──────────────────────────────────────────────────────────
# 奖励函数（奖励模型，全局共享单份）
# ──────────────────────────────────────────────────────────
# 只在 local_rank=0 的进程加载奖励模型（device_map="auto"）。
# reward_fn 被调用时：
#   1. 各进程用 dist.gather_object 把自己的文本发给 rank 0
#   2. rank 0 统一打分
#   3. 用 dist.scatter_object_list 把各自的分数发回
# 单进程运行时退化为直接打分。
# ──────────────────────────────────────────────────────────
def make_reward_fn(reward_model_path: str, max_length: int = 2048):
    import torch.distributed as dist

    _local_rank = int(os.environ.get("LOCAL_RANK", 0))
    _is_main = (_local_rank == 0)

    rm_tok = AutoTokenizer.from_pretrained(reward_model_path, trust_remote_code=True)

    if _is_main:
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            torch_dtype=torch.bfloat16,
            num_labels=1,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
    else:
        rm_model = None

    def _score(texts: list) -> list:
        infer_device = next(rm_model.parameters()).device
        scores = []
        for text in texts:
            if rm_tok.bos_token and text.startswith(rm_tok.bos_token):
                text = text[len(rm_tok.bos_token):]
            inputs = rm_tok(text, return_tensors="pt", truncation=True,
                            max_length=max_length, padding=False).to(infer_device)
            with torch.no_grad():
                scores.append(rm_model(**inputs).logits[0][0].float().item())
        return scores

    def reward_fn(prompts, completions, **kwargs):
        texts = [p + c for p, c in zip(prompts, completions)]

        if not dist.is_initialized():
            return _score(texts)

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # 1. 各进程把自己的文本 gather 到 rank 0
        gathered = [None] * world_size if _is_main else None
        dist.gather_object(texts, gathered, dst=0)

        # 2. rank 0 统一打分，按进程切块
        if _is_main:
            flat = [t for chunk in gathered for t in chunk]
            all_scores = _score(flat)
            n = len(texts)
            chunks = [all_scores[i * n:(i + 1) * n] for i in range(world_size)]
        else:
            chunks = None

        # 3. scatter 回各进程
        out = [None]
        dist.scatter_object_list(out, chunks, src=0)
        return out[0]

    return reward_fn


# ──────────────────────────────────────────────────────────
# 主函数 accelerate launch --num_processes 8 train.py  --reward_model_path ./model2
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3 GRPO 对话问答强化训练（基于 TRL）")

    # ── 模型与数据 ──
    parser.add_argument("--model_path",        type=str, default="./model")
    parser.add_argument("--data_path",         type=str, default="./rlaif.jsonl")
    parser.add_argument("--save_dir",          type=str, default="../out/grpo_qwen3")
    parser.add_argument("--reward_model_path", type=str, required=True,
                        help="奖励模型路径（AutoModelForSequenceClassification）")


    # ── 序列长度 ──
    parser.add_argument("--max_gen_len",    type=int, default=1024,
                        help="生成最大长度，显存不足时调小")

    # ── GRPO 超参 ──
    parser.add_argument("--epochs",             type=int,   default=1)
    parser.add_argument("--num_generations",    type=int,   default=8,
                        help="每个 prompt 生成 N 条回复，需满足 (batch_size×accumulation_steps) % num_generations == 0")
    parser.add_argument("--batch_size",         type=int,   default=1,
                        help="per_device_train_batch_size（prompts 数）")
    parser.add_argument("--accumulation_steps", type=int,   default=8)
    parser.add_argument("--learning_rate",      type=float, default=1e-6)
    parser.add_argument("--grad_clip",          type=float, default=1.0)
    parser.add_argument("--beta",               type=float, default=0.01,
                        help="KL 惩罚系数")
    parser.add_argument("--clip_epsilon",       type=float, default=0.2,
                        help="GRPO ratio 裁剪范围")
    parser.add_argument("--temperature",        type=float, default=0.8)

    # ── 硬件 ──
    parser.add_argument("--dtype",       type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_workers", type=int, default=2)

    # ── 日志与保存 ──
    parser.add_argument("--log_interval",  type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--use_wandb",     action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Qwen3-GRPO")

    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # ── 1. Tokenizer ──
    print(f"加载 Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. 数据集 ──
    dataset = build_dataset(args.data_path, tokenizer)

    # ── 3. 模型 ──
    print(f"加载模型: {args.model_path}  (local_rank={local_rank})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map={"": local_rank},
    )
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # ── 4. 奖励函数 ──
    reward_fn = make_reward_fn(args.reward_model_path)

    # ── 5. GRPOConfig ──
    grpo_config = GRPOConfig(
        output_dir=args.save_dir,
        # 生成参数
        max_completion_length=args.max_gen_len,        # 防止被截断，显式设置
        # GRPO 核心
        num_generations=args.num_generations,
        beta=args.beta,
        epsilon=args.clip_epsilon,

        temperature=args.temperature,

        # 训练参数
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.grad_clip,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",

        # 精度与内存
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.num_workers,

        # 日志与保存
        logging_steps=args.log_interval,
        save_steps=args.save_interval,
        save_total_limit=3,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"qwen3-grpo-lr{args.learning_rate}-beta{args.beta}",
    )

    # ── 6. GRPOTrainer ──
    # dataset 中的 "context_ref" / "question" 列自动作为 kwargs 传入 reward_fn
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # ── 7. 训练 ──
    print("开始 GRPO 训练...")
    trainer.train()

    # ── 8. 保存 ──
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"模型已保存至: {args.save_dir}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
