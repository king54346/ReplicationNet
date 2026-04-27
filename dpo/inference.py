"""
DPO 模型推理脚本

用法：
  # 加载 DPO 训练后保存的模型目录
  python inference.py --model_path ../out/dpo_qwen3

  # 加载原始基础模型
  python inference.py --model_path Qwen/Qwen3-0.6B

  # 单轮推理
  python inference.py --model_path ../out/dpo_qwen3 --prompt "你好"

  # 开启 thinking 模式
  python inference.py --model_path ../out/dpo_qwen3 --thinking
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
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.thinking,
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


def run_interactive(model, tokenizer, args):
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
    messages = [{"role": "user", "content": args.prompt}]
    response = generate(model, tokenizer, messages, args)
    print(f"问: {args.prompt}")
    print(f"答: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO 模型推理")
    parser.add_argument("--model_path", type=str, default="../out/dpo_qwen3",
                        help="模型目录（trainer.save_model 输出 或 HF Hub ID）")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--prompt", type=str, default=None,
                        help="单轮 prompt，不填则进入交互模式")
    parser.add_argument("--max_new_tokens",      type=int,   default=512)
    parser.add_argument("--temperature",         type=float, default=0.7)
    parser.add_argument("--top_p",               type=float, default=0.9)
    parser.add_argument("--repetition_penalty",  type=float, default=1.1)
    parser.add_argument("--thinking", action="store_true",
                        help="开启 Qwen3 thinking 模式")
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model, tokenizer = load_model(args.model_path, dtype)

    if args.prompt:
        run_single(model, tokenizer, args)
    else:
        run_interactive(model, tokenizer, args)
