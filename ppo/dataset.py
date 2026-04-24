from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    对话后处理：清理模板渲染后多余的空 <think> 块。

    特点：
    - 针对带 CoT（chain-of-thought）格式的模型，apply_chat_template 有时会
      渲染出 "<think>\n\n</think>\n\n" 这样的空思考块占位符。
    - 大部分情况下（概率 1 - empty_think_ratio = 95%）直接删除该空块，
      防止模型学到"无意义思考"的坏习惯。
    - 保留少量空思考块（empty_think_ratio = 5%），让模型也能处理该边界情况。
    """
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content

class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        # 保留 bos_id / eos_id 以兼容未来可能的 mask 扩展
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        从对话列表中分离 prompt（上文）和 answer（参考答案）。

        处理逻辑：
        1. 按奇偶索引为每条消息分配 user/assistant 角色。
        2. 记录最后一条消息内容为 answer（即本轮期望的参考回答）。
        3. 用除最后一条之外的消息渲染 prompt，并开启 add_generation_prompt=True，
           使模板在末尾自动追加"assistant 开始回复"的引导标记。
        4. RL actor 收到 prompt 后进行 rollout，生成的回复与 answer 对比打分。
        """
        messages = []
        answer = ""
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
            answer = turn["content"]  # 持续更新，最终保留最后一条 assistant 内容
        # messages[:-1]：去掉最后一条 assistant 回复，只保留上下文
        # add_generation_prompt=True：在末尾追加续写引导 token，告诉模型"现在开始生成"
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = post_processing_chat(prompt)
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        # 返回原始字符串，不做 tokenize，由 RL trainer 在线处理
        prompt, answer = self.create_chat_prompt(sample["conversations"])

        return {"prompt": prompt, "answer": answer}

if __name__ == '__main__':
    # 展示数据格式
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("./model")
    dataset = RLAIFDataset("rlaif.jsonl", tokenizer, max_length=512)
    # {'conversations': [
    # {'role': 'user', 'content': '基于以下角色信息完成一段对话\n张明（30岁），一位IT公司的销售经理。外表干练，擅长沟通。最近他需要协助生产部门促销一批新的智能家居产品，但是他对于这些新产品的技术方面并不够熟悉。\n刘琳（25岁），一名来自生产部门的研发工程师。外表看起来有些内向，但实则是一名技术精湛的工程师。最近生产的智能家居产品是她和她的团队开发的。'},
    # {'role': 'assistant', 'content': '张明：嗨，刘琳，我听说智能家居产品已经进入了我们公司，我真的很想能够了解一下这些产品的技术方面，以便于在销售中向客户传递正确的信息。\n刘琳：嗨，张明，很高兴为您介绍。这些智能家居产品基于物联网技术开发，并且已经通过了一系列的测试，具有很高的品质保证，其中包括智能插座、智能门锁等产品，您想了解哪些方面？\n张明：首先，我想知道产品的使用前提是什么，例如，在产品安装之前客户需要做哪些准备，以及在使用产品时需要注意哪些事项？\n刘琳：客户使用产品之前需要保证有一个稳定的Wi-Fi网络，因为产品只能够通过Wi-Fi网络进行连接和控制。此外，我们提供了一份详细的使用说明书，其中包含了如何安装和使用产品的所有步骤。\n张明：我也想问一下产品的安全性能如何，客户的隐私是否会被泄露？\n刘琳：我们的产品具有完善的安全措施，可以保证客户的隐私不会泄露。这些措施包括数据加密技术、防火墙以及实时的安全更新等。\n张明：听起来很不错，但是我想问一下，如果客户在使用产品时遇到问题，我们应该如何解决？\n刘琳：我们提供了在线客服服务以及售后服务，如果客户在使用过程中遇到问题，可以随时和我们取得联系，我们会在第一时间为客户解决问题。\n张明：感谢您对我的解答，我对这些产品有了更全面的了解，并且可以更好地向客户介绍这些产品了。\n刘琳：没关系，很高兴能够为您提供帮助。'},
    # {'role': 'user', 'content': '基于以上对话提出一个问题。'},
    # {'role': 'assistant', 'content': '这些智能家居产品需要哪些前提条件才能够使用？'},
    # {'role': 'user', 'content': '请回答这个问题。'}, {'role': 'assistant', 'content': ''}
    # ]}
    print(dataset.samples[0])
    # 验证 bos_id 在实际渲染序列里能被匹配到
    sample = dataset.samples[0]