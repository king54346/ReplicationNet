import os
import sys

from dataset import RLAIFDataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    Qwen2ForCausalLM,  # Qwen3 底层继承自 Qwen2 架构
)

warnings.filterwarnings("ignore")

# python ppo_qwen3.py \
#     --model_path /path/to/Qwen3-4B \
#     --reward_model_path /path/to/internlm2-1_8b-reward \
#     --data_path ./rlaif-mini.jsonl \
#     --save_dir ./out \
#     --batch_size 1 \
#     --accumulation_steps 4
#  torchrun --nproc_per_node=2 ppo_qwen3.py 。。。。
# ==========工具函数（替换原 trainer_utils 的依赖）==========

def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def Logger(msg):
    if is_main_process():
        print(msg, flush=True)


def setup_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed_mode():
    """初始化分布式训练，返回 local_rank"""
    local_rank = 0
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        Logger(f"分布式训练初始化完成，local_rank={local_rank}, world_size={dist.get_world_size()}")
    return local_rank


class SkipBatchSampler:
    """跳过前 skip_batches 个 batch，用于从 checkpoint 续训"""
    def __init__(self, sampler, batch_size, skip_batches):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
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
        total = len(self.sampler) // self.batch_size
        return max(0, total - self.skip_batches)


def lm_checkpoint(
    weight,
    model=None,
    optimizer=None,
    epoch=None,
    step=None,
    wandb=None,
    save_dir="./checkpoints",
    scheduler=None,
    critic_model=None,
    critic_optimizer=None,
    critic_scheduler=None,
):
    """
    保存或加载训练 checkpoint。
    - model=None 时为读取模式，返回 checkpoint dict 或 None
    - model!=None 时为写入模式
    """
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = os.path.join(save_dir, f"{weight}_ckp.pt")

    # 读取模式
    if model is None:
        if os.path.exists(ckp_path):
            Logger(f"加载 checkpoint: {ckp_path}")
            return torch.load(ckp_path, map_location="cpu")
        else:
            Logger(f"未找到 checkpoint: {ckp_path}，从头开始训练")
            return None

    # 写入模式
    actor_state = (
        model.module.state_dict()
        if isinstance(model, DistributedDataParallel)
        else model.state_dict()
    )
    critic_state = (
        critic_model.module.state_dict()
        if isinstance(critic_model, DistributedDataParallel)
        else critic_model.state_dict()
    ) if critic_model is not None else None

    ckp = {
        "model": actor_state,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "critic_model": critic_state,
        "critic_optimizer": critic_optimizer.state_dict() if critic_optimizer else None,
        "critic_scheduler": critic_scheduler.state_dict() if critic_scheduler else None,
        "epoch": epoch,
        "step": step,
        "wandb_id": wandb.id if wandb is not None else None,
    }
    torch.save(ckp, ckp_path)
    Logger(f"Checkpoint 已保存: {ckp_path}")

# ==========Critic Model==========

class CriticModel(Qwen2ForCausalLM):
    """
    基于 Qwen3（Qwen2 架构）的 Critic 模型。
    在语言模型主体上加一个线性价值头，输出每个 token 位置的状态价值估计。
    """
    def __init__(self, config):
        super().__init__(config)
        # 价值头：hidden_size -> 1
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
        # 初始化为接近 0，避免训练初期价值估计过大
        nn.init.normal_(self.value_head.weight, std=0.01)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 只取 transformer 主体的 hidden states，不要 lm_head
        kwargs.pop("labels", None)  # Critic 不需要 labels
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        hidden_states = outputs.last_hidden_state  # [B, L, hidden_size]
        values = self.value_head(hidden_states).squeeze(-1)  # [B, L]
        return values


# ==========奖励计算==========

def calculate_rewards(prompts, responses, reward_model, reward_tokenizer, args):
    """
    计算每条生成回复的奖励分数。
    奖励 = 格式奖励（reasoning模式）+ Reward模型打分
    """

    def reasoning_model_reward(rewards):
        """推理模型格式奖励：鼓励模型输出 <think>...</think><answer>...</answer> 结构"""
        pattern  = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        format_rewards = []
        for response in responses:
            if re.match(pattern, response, re.S) or re.match(pattern2, response, re.S):
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # 每个标签各 +0.25，鼓励模型输出完整标签
        def mark_num(text):
            reward = 0.0
            for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
                if text.count(tag) == 1:
                    reward += 0.25
            return reward

        mark_rewards = [mark_num(r) for r in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)

    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # Reward 模型打分
    with torch.no_grad():
        reward_model_scores = []
        scale = 30.0

        for prompt, response in zip(prompts, responses):
            # 从渲染好的 prompt 字符串中还原 messages 列表
            # Qwen3 ChatML 格式：<|im_start|>role\ncontent<|im_end|>
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [
                {"role": role, "content": content.strip()}
                for role, content in matches
            ]

            # 把生成的 response 拼到 messages 末尾
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)
            score = float(max(min(score, scale), -scale))

            # reasoning 模式：单独对 <answer> 内容打分，加权组合
            if args.reasoning == 1:
                answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    tmp_chat_ans = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat_ans)
                    answer_score = float(max(min(answer_score, scale), -scale))
                    score = score * 0.4 + answer_score * 0.6

            reward_model_scores.append(score)

        rewards += torch.tensor(reward_model_scores, device=args.device)

    return rewards


# ==========PPO 训练一个 Epoch==========

def ppo_train_epoch(
    epoch,
    loader,
    iters,
    actor_model,
    old_actor_model,
    ref_model,
    critic_model,
    tokenizer,
    actor_optimizer,
    critic_optimizer,
    actor_scheduler,
    critic_scheduler,
    reward_model,
    reward_tokenizer,
    args,
    start_step=0,
    wandb=None,
):
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]

        # ——— Step 1: tokenize prompt ———
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
        ).to(args.device)

        # 每条 prompt 实际有效 token 数（非 pad 的）
        prompt_lengths = enc.attention_mask.sum(dim=1)  # [B]

        # ——— Step 2: Actor 生成 response ———
        with torch.no_grad():
            model_for_gen = (
                actor_model.module
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model
            )
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        # gen_out: [B, prompt_len + gen_len]

        # 解码 response 部分（去掉 prompt）
        responses_text = [
            tokenizer.decode(
                gen_out[i, prompt_lengths[i]:],
                skip_special_tokens=True
            )
            for i in range(len(prompts))
        ]

        # ——— Step 3: 计算 Reward ———
        rewards = calculate_rewards(
            prompts, responses_text, reward_model, reward_tokenizer, args
        )  # [B]

        # ——— Step 4: Critic 价值估计 ———
        # full_mask: 标记哪些位置是有效 token（非 pad）
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, L]
        value_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, L]

        # 取每条序列最后一个有效 token 的价值作为 V(s)
        last_indices = full_mask.sum(dim=1) - 1                                # [B]
        values = value_seq[torch.arange(len(last_indices)), last_indices]      # [B]

        # Advantage = R - V(s)，detach 让 Critic 梯度独立
        advantages = rewards.to(values.dtype) - values.detach()  # [B]

        # ——— Step 5: Actor 当前 log 概率 ———
        logits = actor_model(
            input_ids=gen_out, attention_mask=full_mask
        ).logits  # [B, L, V]

        labels = gen_out[:, 1:].clone()  # 预测目标：向右移一位

        # 用 cross_entropy(reduction='none') 代替 log_softmax+gather
        # 不实例化完整 [B, L-1, V] softmax 分布，显存占用更低
        B, L, V = logits.shape
        logp_tokens = -F.cross_entropy(
            logits[:, :-1, :].reshape(-1, V),
            labels.reshape(-1),
            reduction="none",
        ).reshape(B, L - 1)  # [B, L-1]
        del logits

        seq_len = gen_out.size(1) - 1
        # resp_mask: True 表示该位置属于 response（不是 prompt）
        resp_mask = (
            torch.arange(seq_len, device=gen_out.device).unsqueeze(0)
            >= prompt_lengths.unsqueeze(1)
        )  # [B, L-1]
        # 同时排除 pad token
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, L-1]

        # 把 response 所有 token 的 log 概率求和 = 该序列的总 log 概率
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  # [B]

        # ——— Step 6: Old Actor 和 Reference log 概率（不需要梯度）———
        with torch.no_grad():
            # device_map="auto" 模型：输入需送到其第一层所在设备，输出再拉回 args.device
            old_dev = next(old_actor_model.parameters()).device
            old_logits = old_actor_model(
                input_ids=gen_out.to(old_dev), attention_mask=full_mask.to(old_dev)
            ).logits.to(args.device)
            old_logp_tokens = -F.cross_entropy(
                old_logits[:, :-1, :].reshape(-1, old_logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).reshape(B, L - 1)
            del old_logits
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]

            ref_dev = next(ref_model.parameters()).device
            ref_logits = ref_model(
                input_ids=gen_out.to(ref_dev), attention_mask=full_mask.to(ref_dev)
            ).logits.to(args.device)
            ref_logp_tokens = -F.cross_entropy(
                ref_logits[:, :-1, :].reshape(-1, ref_logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).reshape(B, L - 1)
            del ref_logits
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        # ——— Step 7: PPO Loss ———
        # KL：actor 与 old_actor 的距离（监控用）
        kl     = (actor_logp - old_logp).mean()
        # KL_ref：actor 与 reference 的距离（惩罚项，防模型遗忘）
        kl_ref = (actor_logp - ref_logp).mean()

        # ratio = π_new / π_old（在 log 空间做减法再 exp）
        ratio = torch.exp(actor_logp - old_logp)  # [B]

        # PPO Clip 损失：取保守（最小）的 surrogate
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Critic MSE 损失
        value_loss = F.mse_loss(values, rewards.to(values.dtype))

        # 总 Loss
        loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref
        loss.backward()

        # ——— Step 8: 梯度更新 ———
        if step % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # ——— Step 9: 日志 ———
        if is_main_process() and (step % args.log_interval == 0 or step == iters):
            # 计算平均 response 长度
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(
                has_eos,
                eos_indices + 1,
                torch.tensor(response_ids.shape[1], device=is_eos.device),
            )
            avg_len = lengths.float().mean().item()

            log_info = (
                f"Epoch: {epoch+1}, Step: {step}/{iters}, "
                f"Actor Loss: {policy_loss.item():.6f}, "
                f"Critic Loss: {value_loss.item():.6f}, "
                f"Reward: {rewards.mean().item():.6f}, "
                f"KL: {kl.item():.6f}, "
                f"KL_ref: {kl_ref.item():.6f}, "
                f"Avg Resp Len: {avg_len:.1f}, "
                f"LR: {actor_optimizer.param_groups[0]['lr']:.2e}"
            )
            Logger(log_info)

            if wandb is not None:
                wandb.log({
                    "actor_loss": policy_loss.item(),
                    "critic_loss": value_loss.item(),
                    "reward": rewards.mean().item(),
                    "kl": kl.item(),
                    "kl_ref": kl_ref.item(),
                    "avg_response_len": avg_len,
                    "actor_lr": actor_optimizer.param_groups[0]["lr"],
                })

        # ——— Step 10: 定期同步 Old Actor ———
        if step % args.update_old_actor_freq == 0:
            actor_sd = (
                actor_model.module.state_dict()
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model.state_dict()
            )
            old_actor_model.load_state_dict(
                {k: v.detach().cpu() for k, v in actor_sd.items()}
            )
            old_actor_model.to(args.device)
            Logger(f"Step {step}: Old Actor 已同步")

        # ——— Step 11: 保存 checkpoint ———
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()

            # 保存 Actor 权重（半精度，节省空间）
            os.makedirs(args.save_dir, exist_ok=True)
            ckp_path = os.path.join(args.save_dir, f"ppo_actor_step{step}.pth")
            actor_sd = (
                actor_model.module.state_dict()
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model.state_dict()
            )
            torch.save({k: v.half() for k, v in actor_sd.items()}, ckp_path)
            Logger(f"Actor 权重已保存: {ckp_path}")

            # 保存完整训练状态（含 Critic，用于续训）
            lm_checkpoint(
                weight="ppo_qwen3",
                model=actor_model,
                optimizer=actor_optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=os.path.join(args.save_dir, "checkpoints"),
                scheduler=actor_scheduler,
                critic_model=critic_model,
                critic_optimizer=critic_optimizer,
                critic_scheduler=critic_scheduler,
            )

            actor_model.train()


# ==========主函数==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-4B PPO 训练")

    # ===== 模型路径 =====
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B",
                        help="Qwen3 模型路径（HuggingFace Hub ID 或本地路径）")
    parser.add_argument("--reward_model_path", type=str, default="internlm/internlm2-1_8b-reward",
                        help="Reward 模型路径")

    # ===== 数据 =====
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl")

    # ===== 保存 =====
    parser.add_argument("--save_dir", type=str, default="../out/ppo_qwen3")

    # ===== 训练基础参数 =====
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="4B 模型显存占用大，建议从 1 开始")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7)
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="梯度累积，等效 batch_size = batch_size * accumulation_steps")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # ===== 序列长度 =====
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Prompt 最大 token 数")
    parser.add_argument("--max_gen_len", type=int, default=1024,
                        help="Response 最大生成 token 数")

    # ===== PPO 超参数 =====
    parser.add_argument("--clip_epsilon", type=float, default=0.1,
                        help="PPO ratio 裁剪范围 [1-ε, 1+ε]")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="Critic 损失系数")
    parser.add_argument("--kl_coef", type=float, default=0.02,
                        help="KL 惩罚系数（防止偏离 Reference 太远）")
    parser.add_argument("--update_old_actor_freq", type=int, default=4,
                        help="每隔多少 step 把 Actor 权重同步到 Old Actor")

    # ===== 推理模式 =====
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1],
                        help="1=推理模型（<think><answer>格式奖励），0=普通模型")

    # ===== 硬件 =====
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--num_workers", type=int, default=1)

    # ===== 日志与续训 =====
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1],
                        help="1=从 checkpoint 续训")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Qwen3-PPO")

    args = parser.parse_args()

    # ========== 1. 初始化分布式 & 随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # ========== 2. 检查 checkpoint ==========
    ckp_data = None
    if args.from_resume == 1:
        ckp_data = lm_checkpoint(
            weight="ppo_qwen3",
            save_dir=os.path.join(args.save_dir, "checkpoints"),
        )

    # ========== 3. 初始化 wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        wandb.init(
            project=args.wandb_project,
            name=f"qwen3-ppo-bs{args.batch_size}-lr{args.learning_rate}",
            id=wandb_id,
            resume="must" if wandb_id else None,
        )

    # ========== 4. 加载 Tokenizer ==========
    Logger(f"加载 Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"   # PPO generate 需要左侧 padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        Logger("pad_token 未设置，已自动设为 eos_token")

    # ========== 5. 加载 Actor 模型 ==========
    Logger(f"加载 Actor 模型: {args.model_path}")
    actor_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=args.device,
    )

    # ========== 6. 加载 Old Actor（重要性采样基准，不训练）==========
    Logger("加载 Old Actor 模型...")
    old_actor_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    ).eval().requires_grad_(False)

    # ========== 7. 加载 Reference 模型（KL 惩罚基准，不训练，固定不变）==========
    Logger("加载 Reference 模型...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    ).eval().requires_grad_(False)

    # ========== 8. 加载 Critic 模型 ==========
    Logger("加载 Critic 模型...")
    critic_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    critic_model = CriticModel.from_pretrained(
        args.model_path,
        config=critic_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,  # value_head 是新加的层，忽略权重不匹配
    ).to(args.device)

    # ========== 9. 加载 Reward 模型（外部预训练，不训练）==========
    Logger(f"加载 Reward 模型: {args.reward_model_path}")

    class SequenceClassificationRewardModel:
        """将 AutoModelForSequenceClassification 包装为兼容 get_score 接口的 reward model。"""
        def __init__(self, model, tokenizer, device):
            self._model = model
            self._tokenizer = tokenizer
            self._device = device

        def get_score(self, tok, messages):
            text = tok.apply_chat_template(messages, tokenize=False)
            if tok.bos_token and text.startswith(tok.bos_token):
                text = text[len(tok.bos_token):]
            inputs = tok(
                text, return_tensors="pt", truncation=True, max_length=4096
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
            return logits[0][0].item()

    _reward_base = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        num_labels=1,
        device_map="auto",
    ).to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_path, trust_remote_code=True
    )
    reward_model = SequenceClassificationRewardModel(_reward_base, reward_tokenizer, args.device)

    # ========== 10. 数据集 & 优化器 ==========
    train_ds = RLAIFDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len + args.max_gen_len,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    actor_optimizer  = optim.AdamW(actor_model.parameters(),  lr=args.learning_rate,        weight_decay=0.01)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate, weight_decay=0.01)

    # 用一个临时 loader 计算 iters 数
    _tmp_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(_tmp_loader)
    total_steps = max(1, (iters // args.accumulation_steps) * args.epochs)

    actor_scheduler  = CosineAnnealingLR(actor_optimizer,  T_max=total_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_steps, eta_min=args.critic_learning_rate / 10)

    # ========== 11. 从 checkpoint 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data["model"])
        critic_model.load_state_dict(ckp_data["critic_model"])
        actor_optimizer.load_state_dict(ckp_data["optimizer"])
        critic_optimizer.load_state_dict(ckp_data["critic_optimizer"])
        actor_scheduler.load_state_dict(ckp_data["scheduler"])
        critic_scheduler.load_state_dict(ckp_data["critic_scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step  = ckp_data.get("step", 0)
        Logger(f"从 checkpoint 恢复：epoch={start_epoch}, step={start_step}")

    # ========== 12. DDP 包装 ==========
    if dist.is_initialized():
        # freqs_cos/freqs_sin 是 RoPE 的位置编码缓存，不参与 DDP 同步
        for m in [actor_model, critic_model]:
            m._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model  = DistributedDataParallel(actor_model,  device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        # old_actor_model.to(args.device)
        ref_model.to(args.device)

    Logger(f"训练开始：epochs={args.epochs}, iters/epoch={iters}, total_steps={total_steps}")

    # ========== 13. 训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # 续训时第一个 epoch 跳过已训练的 step
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
            ppo_train_epoch(
                epoch, loader, len(loader) + start_step,
                actor_model, old_actor_model, ref_model, critic_model,
                tokenizer, actor_optimizer, critic_optimizer,
                actor_scheduler, critic_scheduler,
                reward_model, reward_tokenizer,
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
            ppo_train_epoch(
                epoch, loader, iters,
                actor_model, old_actor_model, ref_model, critic_model,
                tokenizer, actor_optimizer, critic_optimizer,
                actor_scheduler, critic_scheduler,
                reward_model, reward_tokenizer,
                args, 0, wandb,
            )

    Logger("训练完成！")