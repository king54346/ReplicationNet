"""
MiniSGL LLM Inference Example
展示基本的推理功能和性能测试
"""

import os
import time
from random import randint, seed

from minisgl.core import SamplingParams
from minisgl.llm import LLM
from transformers import AutoTokenizer

_llm = None
_tokenizer = None
def get_llm(model_path):
    """获取或创建 LLM 实例（单例模式）"""
    global _llm, _tokenizer
    
    if _llm is None:
        print("🔧 Loading model...")
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _llm = LLM(
            model_path,
            max_seq_len_override=4096,
            max_extend_tokens=16384,
            cuda_graph_max_bs=256,
        )
        print("✅ Model loaded\n")
    
    return _llm, _tokenizer


def simple_chat_example():
    """简单的对话示例"""
    print("="*60)
    print("💬 Simple Chat Example")
    print("="*60 + "\n")
    
    model_path = os.path.expanduser("/home/user/demo/AsLive-main/LLM/qwen/Qwen3-4B/")
    
    # 加载 tokenizer 和模型
    llm, tokenizer = get_llm(model_path)
    
    # 准备对话
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "explain what is attention mechanism in transformers",
    ]
    
    # 应用 chat template
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    # 生成回复
    print("🚀 Generating responses...\n")
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # 打印结果
    for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
        print(f"{'─'*60}")
        print(f"Query {i}: {prompt}")
        print(f"{'─'*60}")
        print(f"Response: {output['text']}")
        print()


def streaming_example():
    """流式输出示例（如果支持）"""
    print("="*60)
    print("🌊 Streaming Example")
    print("="*60 + "\n")
    
    model_path = os.path.expanduser("/home/user/demo/AsLive-main/LLM/qwen/Qwen3-4B/")
    llm, tokenizer = get_llm(model_path)
    
    prompt = "Write a short story about a robot learning to paint."
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=512,
        top_p=0.9,
    )
    
    print(f"Prompt: {prompt}\n")
    print("Response:", end=" ", flush=True)
    
    # 注意: 如果 minisgl 不支持流式，这里会一次性输出
    outputs = llm.generate([formatted_prompt], sampling_params)
    
    for output in outputs:
        print(output['text'])
    print()


def batch_comparison():
    """对比不同批次大小的性能"""
    print("\n" + "="*60)
    print("📊 Batch Size Comparison")
    print("="*60 + "\n")
    
    model_path = os.path.expanduser("/home/user/demo/AsLive-main/LLM/qwen/Qwen3-4B/")
    
    batch_sizes = [16, 32, 64, 128, 256]
    results = []
    
    for batch_size in batch_sizes:
        print(f"Testing batch_size={batch_size}...", end=" ", flush=True)
        
        llm = LLM(
            model_path,
            max_seq_len_override=4096,
            max_extend_tokens=16384,
            cuda_graph_max_bs=256,
        )
        
        seed(42)
        prompt_token_ids = [
            [randint(0, 10000) for _ in range(512)]
            for _ in range(batch_size)
        ]
        
        sampling_params = [
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=256)
            for _ in range(batch_size)
        ]
        
        # Warmup
        llm.generate(["W"], SamplingParams(max_tokens=5))
        
        start = time.time()
        outputs = llm.generate(prompt_token_ids, sampling_params)
        elapsed = time.time() - start
        
        total_tokens = batch_size * 256
        throughput = total_tokens / elapsed
        
        results.append({
            'batch_size': batch_size,
            'throughput': throughput,
            'time': elapsed,
        })
        
        print(f"{throughput:.2f} tok/s")
        
        # 清理
        del llm
        import torch
        torch.cuda.empty_cache()
    
    # 打印对比表格
    print(f"\n{'='*60}")
    print(f"{'Batch Size':>15} {'Throughput':>20} {'Time':>15}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['batch_size']:>15} {r['throughput']:>15.2f} tok/s {r['time']:>12.2f}s")
    print(f"{'='*60}\n")


def main():
    """主函数 - 运行所有示例"""
    
    # 1. 简单对话示例
    simple_chat_example()
    
    streaming_example()  # 取消注释以运行
    
    # batch_comparison()  # 取消注释以运行


if __name__ == "__main__":
    main()