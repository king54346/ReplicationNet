"""
GRPO 对话问答模型推理脚本

用法：
  # 交互模式（多轮对话）
  python inference.py --model_path ../out/grpo_qwen3

  # 单轮推理
  python inference.py --model_path ../out/grpo_qwen3 --prompt "Wi-Fi 对智能家居有什么作用？"

  # Demo 模式：模拟训练数据格式（背景对话 → 提问 → 回答）
  python inference.py --model_path ../out/grpo_qwen3 --demo

  # 对比基础模型
  python inference.py --model_path ./model --prompt "..."

  # 开启 thinking 模式
  python inference.py --model_path ../out/grpo_qwen3 --thinking
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, dtype: torch.dtype):
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    param_b = sum(p.numel() for p in model.parameters()) / 1e9
    devices = {str(p.device) for p in model.parameters()}
    print(f"参数量: {param_b:.2f}B  设备: {devices}\n")
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, messages: list[dict], args) -> str:
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.thinking,
        )
    except TypeError:
        # 部分版本不支持 enable_thinking 参数
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(text, return_tensors="pt")
    input_ids      = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    out_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_ids = out_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────
# Demo 模式：复现训练数据格式
#   Step1  user   → 提供背景对话
#   Step2  asst   → 给出完整对话内容（可手动输入或使用内置示例）
#   Step3  user   → "基于以上对话提出一个问题。"
#   Step4  asst   → 模型生成问题
#   Step5  user   → "请回答这个问题。"
#   Step6  asst   → 模型生成答案  ← 这是 GRPO 训练的目标轮次
# ──────────────────────────────────────────────────────────
DEMO_CONTEXT = (
    "张明：嗨，刘琳，我听说智能家居产品已经进入了我们公司，我真的很想能够了解一下这些产品的"
    "技术方面，以便于在销售中向客户传递正确的信息。\n"
    "刘琳：嗨，张明，很高兴为您介绍。这些智能家居产品基于物联网技术开发，并且已经通过了一系列"
    "的测试，具有很高的品质保证，其中包括智能插座、智能门锁等产品，您想了解哪些方面？\n"
    "张明：首先，我想知道产品的使用前提是什么，例如，在产品安装之前客户需要做哪些准备？\n"
    "刘琳：客户使用产品之前需要保证有一个稳定的Wi-Fi网络，因为产品只能够通过Wi-Fi网络进行"
    "连接和控制。此外，我们提供了一份详细的使用说明书，其中包含了如何安装和使用产品的所有步骤。"
)

DEMO_BACKGROUND = (
    "基于以下角色信息完成一段对话\n"
    "张明（30岁），一位IT公司的销售经理。外表干练，擅长沟通。最近他需要协助生产部门促销一批"
    "新的智能家居产品，但是他对于这些新产品的技术方面并不够熟悉。\n"
    "刘琳（25岁），一名来自生产部门的研发工程师。外表看起来有些内向，但实则是一名技术精湛的"
    "工程师。最近生产的智能家居产品是她和她的团队开发的。"
)


def run_demo(model, tokenizer, args):
    """模拟训练数据的多轮格式，展示模型在该场景下的能力。"""
    print("=" * 50)
    print("Demo 模式：复现训练数据格式")
    print("=" * 50)

    # 构造与训练数据完全一致的 messages（除最后一条 assistant 外）
    messages = [
        {"role": "user",      "content": DEMO_BACKGROUND},
        {"role": "assistant", "content": DEMO_CONTEXT},
        {"role": "user",      "content": "基于以上对话提出一个问题。"},
    ]

    print(f"\n[背景]\n{DEMO_BACKGROUND}\n")
    print(f"[对话内容]\n{DEMO_CONTEXT}\n")
    print("[Step 1] 让模型提出一个问题...")
    question = generate(model, tokenizer, messages, args)
    print(f"模型提问: {question}\n")

    messages.append({"role": "assistant", "content": question})
    messages.append({"role": "user",      "content": "请回答这个问题。"})

    print("[Step 2] 让模型回答这个问题...")
    answer = generate(model, tokenizer, messages, args)
    print(f"模型回答: {answer}\n")


def run_interactive(model, tokenizer, args):
    """多轮交互对话。"""
    print("交互模式  |  clear=清空历史  quit=退出")
    print("-" * 40)
    history = []

    while True:
        try:
            user_input = input("用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "clear":
            history.clear()
            print("(历史已清空)")
            continue

        history.append({"role": "user", "content": user_input})
        response = generate(model, tokenizer, history, args)
        history.append({"role": "assistant", "content": response})
        print(f"助手: {response}\n")


def run_single(model, tokenizer, args):
    """单轮推理。"""
    messages = [{"role": "user", "content": args.prompt}]
    response = generate(model, tokenizer, messages, args)
    print(f"问: {args.prompt}")
    print(f"答: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO 对话问答模型推理")
    parser.add_argument("--model_path", type=str, default="../out/grpo_qwen3")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--prompt", type=str, default=None,
                        help="单轮 prompt，不填则进入交互模式")
    parser.add_argument("--demo", action="store_true",
                        help="Demo 模式：复现训练数据格式（背景对话→提问→回答）")
    parser.add_argument("--max_new_tokens",     type=int,   default=512)
    parser.add_argument("--temperature",        type=float, default=0.7)
    parser.add_argument("--top_p",              type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--thinking", action="store_true",
                        help="开启 Qwen3 thinking 模式")
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    model, tokenizer = load_model(args.model_path, dtype)

    if args.demo:
        run_demo(model, tokenizer, args)
    elif args.prompt:
        run_single(model, tokenizer, args)
    else:
        run_interactive(model, tokenizer, args)
