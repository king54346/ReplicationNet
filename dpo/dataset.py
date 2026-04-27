from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset


def post_processing_chat(text):
    return text.strip(' ')
# ──────────────────────────────────────────────────────────────────────────────
# 3. DPODataset —— 直接偏好优化（Direct Preference Optimization）数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：让模型学会"偏好好回答、远离坏回答"，使输出更符合人类偏好
# 数据格式：{"chosen": [{role, content}...], "rejected": [{role, content}...]}
#   - chosen：人类标注的更优回答对话
#   - rejected：人类标注的较差回答对话
# 训练特点：
#   - 每条样本同时返回 chosen 和 rejected 两份 tokenized 序列，
#     训练时 DPO loss 会最大化 chosen 回复的对数似然、最小化 rejected 的。
#   - loss_mask 的设计与 SFT 一致：只有 assistant 回复部分为 1，
#     其余为 0，保证对比信号仅来自模型的实际输出部分。
#   - 采用"错位"方式构造输入输出对：x 取 [:-1]，y 取 [1:]，
#     即 x[t] 预测 y[t] = input[t+1]，标准自回归格式。
#   - mask 同样错位取 [1:]，与 y 对齐，方便在训练时直接做 masked loss。
#   - max_length 默认 4096，比 SFT 更长，因为 DPO 数据通常包含完整对话上下文。
# ──────────────────────────────────────────────────────────────────────────────
class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 直接 tokenize assistant 块的起始标记
        # Qwen3 chat template 固定格式：<|im_start|>assistant\n
        # '<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n你好！<|im_end|>\n'
        # 不管 thinking 开不开，这个前缀都存在
        self.bos_id = tokenizer(
            "<|im_start|>assistant\n",
            add_special_tokens=False
        ).input_ids

        # assistant 块结束标记：<|im_end|>\n
        self.eos_id = tokenizer(
            "<|im_end|>\n",
            add_special_tokens=False
        ).input_ids

        print(f"bos_id tokens: {tokenizer.convert_ids_to_tokens(self.bos_id)}")
        print(f"eos_id tokens: {tokenizer.convert_ids_to_tokens(self.eos_id)}")

        # thinking 模式下 assistant 内容前有 <think>\n\n</think>\n\n
        # 需要跳过这段，只对真正的回复内容做 mask
        self.think_start_id = tokenizer(
            "<think>",
            add_special_tokens=False
        ).input_ids
        self.think_end_id = tokenizer(
            "</think>\n\n",
            add_special_tokens=False
        ).input_ids

        self.samples = load_dataset("json", data_files=data_path, split="train")
        print(f"DPODataset 加载完成，共 {len(self.samples)} 条样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample["chosen"]  # 优质回答对话列表
        rejected = sample["rejected"]  # 劣质回答对话列表

        # Step 1：渲染为字符串
        chosen_prompt = post_processing_chat(
            self.tokenizer.apply_chat_template(
                chosen,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,  # ← 加这行
            )
        )
        rejected_prompt = post_processing_chat(
            self.tokenizer.apply_chat_template(
                rejected,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,  # ← 加这行
            )
        )

        # Step 2：tokenize + padding 到 max_length
        chosen_enc = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding="max_length")
        rejected_enc = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length,
                                      padding="max_length")

        chosen_input_ids = chosen_enc["input_ids"]
        rejected_input_ids = rejected_enc["input_ids"]

        # Step 3：生成 loss mask（只对 assistant 回复区间置 1）
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # Step 4：自回归对齐，x=[:-1] 输入，y=[1:] 目标，mask=[1:] 与 y 对齐
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        # attention_mask：非 padding 位置为 1
        attention_mask_chosen = (x_chosen != self.padding).long()
        attention_mask_rejected = (x_rejected != self.padding).long()

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
            "attention_mask_chosen": attention_mask_chosen,
            "attention_mask_rejected": attention_mask_rejected,
        }

    def generate_loss_mask(self, input_ids):
        """
        生成 DPO 训练所需的 loss mask（0/1 二值序列）。

        与 SFTDataset.generate_labels 逻辑完全相同，区别在于：
        - SFT 返回的是具体的 token id（用于 CE loss）
        - DPO 返回的是 0/1 掩码（用于 masked 对数似然计算）
        算法：扫描 bos_id → 找到 eos_id → 区间内置 1，其余置 0。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将 assistant 回复（含 EOS）区间的 mask 置 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

if __name__ == '__main__':
    # 展示数据格式
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("./model")
    dataset = DPODataset("dpo.jsonl", tokenizer, max_length=512)
    # dataset[i]  {
    #   "x_chosen":              tensor([...], shape=[511])  # token ids
    #   "y_chosen":              tensor([...], shape=[511])  # 自回归错位
    #   "mask_chosen":           tensor([...], shape=[511])  # 0/1 mask
    #   "x_rejected":            tensor([...], shape=[511])
    #   "y_rejected":            tensor([...], shape=[511])
    #   "mask_rejected":         tensor([...], shape=[511])
    #   "attention_mask_chosen":  tensor([...], shape=[511])
    #   "attention_mask_rejected":tensor([...], shape=[511])
    # }
    # 举例：
    # x_chosen:               模型输入序列（chosen，去掉最后一个token）
    #                         → [im_start, user, \n, 是~还有你嘛, im_end, \n, im_start, asst, \n, 你个傻逼..., im_end]
    #
    # y_chosen:               预测目标（chosen，去掉第一个token，左移一位）
    #                         → [user, \n, 是~还有你嘛, im_end, \n, im_start, asst, \n, 你个傻逼..., im_end, pad]
    #
    # mask_chosen:            loss mask（只有 assistant content 是 1）
    #                         → [0,0,0,...,0, 1,1,1,1,1,1,1,1, 0]
    #                                         ↑"你个傻逼..."↑
    #
    # x_rejected:             模型输入序列（rejected）
    #                         → [im_start, user, \n, 是~还有你嘛, im_end, \n, im_start, asst, \n, 我是AI..., im_end]
    #
    # y_rejected:             预测目标（rejected）
    #                         → [user, \n, 是~还有你嘛, im_end, \n, im_start, asst, \n, 我是AI..., im_end, pad]
    #
    # mask_rejected:          loss mask（只有 assistant content 是 1）
    #                         → [0,0,0,...,0, 1,1,1,1,1,1,1,1,1, 0]
    #                                         ↑"我是AI..."↑
    #
    # attention_mask_chosen:  真实token位置是1，pad位置是0（基于 x_chosen）
    #                         → [1,1,1,...,1,1,0,0]
    #
    # attention_mask_rejected:真实token位置是1，pad位置是0（基于 x_rejected）
    #                         → [1,1,1,...,1,1,0,0,0]  ← rejected 回答更长，pad 更少
    #
    # x_chosen chosen 对话的输入 token ids，shape=[511]
    # y_chosen chosen 对话的目标 token ids（x 错位一位），shape=[511] 本质是根据前文预测下一个词
    # 例如 输入位置i 看到 x[i] 预测 y[i]
    # mask_chosen chosen 的 loss mask，只有 assistant 回复区间为 1，其余（user/system/padding）为 0
    # x_rejected rejected 对话的输入 token ids
    # y_rejected rejected 对话的目标 token ids
    # mask_rejected rejected 的 loss mask，同上，只标记 assistant 回复区间
    # attention_mask_chosen chosen 的注意力 mask，非 padding 位置为 1，padding 位置为 0
    # attention_mask_rejected rejected 的注意力 mask，同上
    print(dataset.samples[0])
    # 验证 bos_id 在实际渲染序列里能被匹配到
    sample = dataset.samples[0]
    text = tokenizer.apply_chat_template(sample["chosen"], tokenize=False,
                                        add_generation_prompt=False, enable_thinking=False)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    mask = dataset.generate_loss_mask(ids + [tokenizer.pad_token_id] * (dataset.max_length - len(ids)))
    print("mask sum:", sum(mask), "total:", len(ids))
    # 把 mask=1 的位置 decode 出来看看是不是 assistant 回复
    assistant_tokens = [tok for tok, m in zip(ids, mask[:len(ids)]) if m == 1]
    print(tokenizer.decode(assistant_tokens))