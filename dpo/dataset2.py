from torch.utils.data import Dataset
import torch
from datasets import load_dataset


def post_processing_chat(text):
    return text.strip()


class DPODataset2(Dataset):
    """
    适配 btfChinese_DPO.jsonl 这类扁平字段格式的 DPO 数据集。

    单条原始样本示例：
    {
        "system": "...",
        "question": "...",
        "chosen": "...",
        "rejected": "..."
    }

    返回字段与 dataset.py 中 DPODataset 保持一致，方便直接复用 train.py。
    """

    def __init__(self, data_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        self.bos_id = tokenizer(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        ).input_ids
        self.eos_id = tokenizer(
            "<|im_end|>\n",
            add_special_tokens=False,
        ).input_ids

        self.samples = load_dataset("json", data_files=data_path, split="train")
        print(f"DPODataset2 加载完成，共 {len(self.samples)} 条样本")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _to_messages(system_text, question_text, answer_text):
        messages = []
        if isinstance(system_text, str) and system_text.strip():
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": question_text or ""})
        messages.append({"role": "assistant", "content": answer_text or ""})
        return messages

    def _build_pair_messages(self, sample):
        # 支持两种格式：
        # 1) 扁平字符串字段：system/question/chosen/rejected
        # 2) 已是对话列表字段：chosen/rejected（兼容旧格式）
        if isinstance(sample.get("chosen"), list) and isinstance(sample.get("rejected"), list):
            return sample["chosen"], sample["rejected"]

        system_text = sample.get("system", "")
        question_text = sample.get("question", "")
        chosen_text = sample.get("chosen", "")
        rejected_text = sample.get("rejected", "")

        chosen_messages = self._to_messages(system_text, question_text, chosen_text)
        rejected_messages = self._to_messages(system_text, question_text, rejected_text)
        return chosen_messages, rejected_messages

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen_messages, rejected_messages = self._build_pair_messages(sample)

        chosen_prompt = post_processing_chat(
            self.tokenizer.apply_chat_template(
                chosen_messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )
        rejected_prompt = post_processing_chat(
            self.tokenizer.apply_chat_template(
                rejected_messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )

        chosen_enc = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        rejected_enc = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        chosen_input_ids = chosen_enc["input_ids"]
        rejected_input_ids = rejected_enc["input_ids"]

        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

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
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i: i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end: end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("./model")
    dataset = DPODataset2("btfChinese_DPO.jsonl", tokenizer, max_length=512)

    print("raw sample:")
    print(dataset.samples[0])

    item = dataset[0]
    print("\nprocessed keys:", list(item.keys()))
    for k, v in item.items():
        print(k, v.shape, v.dtype)

