"""
CTC Inference Script - Enhanced
支持单个文件和批量推理
"""
import os
import torch
from tokenizers import Tokenizer
from model import CTCASR
from tqdm import tqdm
import argparse


def load_model(checkpoint_path, tokenizer_file, device='cuda'):
    """加载模型"""
    tokenizer = Tokenizer.from_file(tokenizer_file)
    vocab_size = tokenizer.get_vocab_size()
    
    model = CTCASR(
        n_mels=80,
        d_model=512,
        nhead=8,
        num_encoder_layers=12,
        dim_feedforward=2048,
        dropout=0.0,
        vocab_size=vocab_size,
        blank_id=0,
        use_cnn_subsampling=True,
        use_spec_augment=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def load_audio(audio_path):
    """加载音频文件"""
    audio_data = torch.load(audio_path, weights_only=False)
    
    # 处理不同格式
    if isinstance(audio_data, dict):
        # 预处理后的.pt文件
        audio = audio_data['audio_features']
        audio_length = audio_data.get('audio_length', audio.size(0))
        text_gt = audio_data.get('text', None)
        return audio, audio_length, text_gt
    else:
        # 直接是tensor
        audio = audio_data
        audio_length = audio.size(0)
        return audio, audio_length, None


def decode_tokens(tokens, tokenizer):
    """Token → 文本"""
    # 移除特殊tokens (0-4: BLANK, PAD, UNK, BOS, EOS)
    filtered = [t for t in tokens if t >= 5]
    if not filtered:
        return ""
    text = tokenizer.decode(filtered)
    return text


def calculate_wer(reference, hypothesis):
    """计算 Character Error Rate"""
    ref = reference.replace(' ', '').strip().lower()
    hyp = hypothesis.replace(' ', '').strip().lower()
    
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    
    return d[-1][-1] / len(ref)


def infer_single(model, audio, audio_length, tokenizer, beam_size=None, device='cuda'):
    """单个样本推理"""
    audio = audio.to(device)
    audio_length = torch.tensor([audio_length], device=device)
    
    with torch.no_grad():
        if beam_size is None or beam_size == 1:
            # Greedy
            decoded = model.decode_greedy(audio.unsqueeze(0), audio_length)
        else:
            # Beam Search
            decoded = model.decode_beam_search(audio.unsqueeze(0), audio_length, beam_size)
    
    tokens = decoded[0]
    text = decode_tokens(tokens, tokenizer)
    
    return text, tokens


def infer_batch(model, audio_list, tokenizer, beam_size=None, device='cuda'):
    """批量推理"""
    results = []
    
    for audio_path in tqdm(audio_list, desc="Inference"):
        try:
            audio, audio_length, text_gt = load_audio(audio_path)
            text_pred, tokens = infer_single(model, audio, audio_length, tokenizer, beam_size, device)
            
            result = {
                'file': audio_path,
                'ground_truth': text_gt,
                'prediction': text_pred,
                'tokens': tokens
            }
            
            if text_gt:
                result['wer'] = calculate_wer(text_gt, text_pred)
            
            results.append(result)
        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")
    
    return results


def print_result(result, show_tokens=False):
    """打印推理结果"""
    print("\n" + "=" * 70)
    print(f"File: {result['file']}")
    if result.get('ground_truth'):
        print(f"Ground Truth: {result['ground_truth']}")
    print(f"Prediction:   {result['prediction']}")
    if 'wer' in result:
        print(f"WER: {result['wer']:.2f}%")
    if show_tokens:
        print(f"Tokens: {result['tokens'][:30]}...")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='CTC Inference')
    
    # 必需参数
    parser.add_argument('--checkpoint', required=True, help='模型checkpoint路径')
    parser.add_argument('--tokenizer', default='tokenizer_ctc.json', help='Tokenizer文件')
    
    # 输入（二选一）
    parser.add_argument('--audio', help='单个音频文件(.pt)')
    parser.add_argument('--audio_list', help='音频文件列表（每行一个路径）')
    parser.add_argument('--dataset_dir', help='数据集目录（配合metadata使用）')
    parser.add_argument('--metadata', help='Metadata文件（train.txt/val.txt/test.txt）')
    
    # 推理参数
    parser.add_argument('--beam_size', type=int, default=10, help='Beam size (1=greedy)')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--show_tokens', action='store_true', help='显示tokens')
    parser.add_argument('--output', help='输出结果到文件')
    
    args = parser.parse_args()
    
    # 检查输入
    if not any([args.audio, args.audio_list, (args.dataset_dir and args.metadata)]):
        parser.error("Must specify --audio, --audio_list, or --dataset_dir + --metadata")
    
    # 加载模型
    print("Loading model...")
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, args.device)
    
    # 准备音频列表
    audio_files = []
    
    if args.audio:
        # 单个文件
        audio_files = [args.audio]
    
    elif args.audio_list:
        # 文件列表
        with open(args.audio_list, 'r') as f:
            audio_files = [line.strip() for line in f if line.strip()]
    
    elif args.dataset_dir and args.metadata:
        # 从metadata加载
        with open(args.metadata, 'r') as f:
            metas = [line.strip().split()[0] for line in f if line.strip()]
        audio_files = [os.path.join(args.dataset_dir, f'{meta}.pt') for meta in metas]
    
    print(f"\nProcessing {len(audio_files)} files...")
    print(f"Beam size: {args.beam_size} ({'greedy' if args.beam_size == 1 else 'beam search'})")
    
    # 推理
    if len(audio_files) == 1:
        # 单个文件
        audio, audio_length, text_gt = load_audio(audio_files[0])
        print(f"\nAudio shape: {audio.shape}")
        print(f"Audio length: {audio_length}")
        if text_gt:
            print(f"Ground truth: {text_gt}")
        
        print("\nInferencing...")
        text_pred, tokens = infer_single(model, audio, audio_length, tokenizer, 
                                        args.beam_size, args.device)
        
        result = {
            'file': audio_files[0],
            'ground_truth': text_gt,
            'prediction': text_pred,
            'tokens': tokens
        }
        if text_gt:
            result['wer'] = calculate_wer(text_gt, text_pred)
        
        print_result(result, args.show_tokens)
    
    else:
        # 批量推理
        results = infer_batch(model, audio_files, tokenizer, args.beam_size, args.device)
        
        # 打印结果
        for result in results[:5]:  # 只显示前5个
            print_result(result, args.show_tokens)
        
        if len(results) > 5:
            print(f"\n... ({len(results) - 5} more results)")
        
        # 统计
        if any('wer' in r for r in results):
            wers = [r['wer'] for r in results if 'wer' in r]
            avg_wer = sum(wers) / len(wers)
            print(f"\n{'=' * 70}")
            print(f"Average WER: {avg_wer:.2f}%")
            print(f"Samples: {len(wers)}")
            print(f"{'=' * 70}")
        
        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"File: {result['file']}\n")
                    if result.get('ground_truth'):
                        f.write(f"GT:   {result['ground_truth']}\n")
                    f.write(f"Pred: {result['prediction']}\n")
                    if 'wer' in result:
                        f.write(f"WER:  {result['wer']:.2f}%\n")
                    f.write("\n")
            print(f"\n✓ Results saved to {args.output}")


if __name__ == '__main__':
    main()