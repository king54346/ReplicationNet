"""
Inference and Evaluation - 智能后处理版本
"""
import os
import argparse
import re
from typing import List, Dict

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

from model import TransformerASR
from dataset import get_dataloaders


class ASRInference:
    """ASR 推理器"""
    
    def __init__(self, model, tokenizer, device, max_len=200):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        
        self.pad_id = tokenizer.token_to_id('[PAD]')
        self.bos_id = tokenizer.token_to_id('[BOS]')
        self.eos_id = tokenizer.token_to_id('[EOS]')
        
        self.model.eval()
    
    @torch.no_grad()
    def greedy_decode(self, audio: torch.Tensor) -> List[int]:
        """贪心解码"""
        memory = self.model.encode(audio, None)
        tgt = torch.tensor([[self.bos_id]], dtype=torch.long, device=self.device)
        
        for _ in range(self.max_len):
            output = self.model.decode(tgt, memory, None, None, None)
            next_token = output[:, -1, :].argmax(dim=-1)
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            
            if next_token.item() == self.eos_id:
                break
        
        tokens = tgt[0].cpu().tolist()[1:]
        if tokens and tokens[-1] == self.eos_id:
            tokens = tokens[:-1]
        
        return tokens
    
    @torch.no_grad()
    def beam_search(self, audio: torch.Tensor, beam_size=5) -> List[int]:
        """Beam Search 解码"""
        memory = self.model.encode(audio, None)
        beams = [(0.0, [self.bos_id])]
        
        for _ in range(self.max_len):
            new_beams = []
            
            for score, seq in beams:
                if seq[-1] == self.eos_id:
                    new_beams.append((score, seq))
                    continue
                
                tgt = torch.tensor([seq], dtype=torch.long, device=self.device)
                output = self.model.decode(tgt, memory, None, None, None)
                
                log_probs = F.log_softmax(output[:, -1, :], dim=-1)[0]
                top_log_probs, top_tokens = log_probs.topk(beam_size)
                
                for log_prob, token in zip(top_log_probs, top_tokens):
                    new_score = score + log_prob.item()
                    new_seq = seq + [token.item()]
                    new_beams.append((new_score, new_seq))
            
            beams = sorted(new_beams, key=lambda x: x[0] / len(x[1]), reverse=True)[:beam_size]
            
            if all(seq[-1] == self.eos_id for _, seq in beams):
                break
        
        best_seq = beams[0][1][1:]
        if best_seq and best_seq[-1] == self.eos_id:
            best_seq = best_seq[:-1]
        
        return best_seq
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Token IDs -> 文本 + 后处理"""
        filtered = [t for t in token_ids if t not in [self.pad_id, self.bos_id, self.eos_id]]
        text = self.tokenizer.decode(filtered)
        text = self.post_process(text)
        return text
    
    def post_process(self, text: str) -> str:
        """
        后处理 - 针对字符级 tokenizer
        如果检测到大量短token，直接去掉所有空格
        """
        text = text.strip()
        if not text:
            return text
        
        tokens = text.split()
        
        # 检测短token比例
        short_count = sum(1 for t in tokens if len(t) <= 3 and t.isalpha())
        
        if short_count >= len(tokens) * 0.6:  # ≥60%是短token
            # 字符级输出，去掉所有空格
            return text.replace(' ', '')
        else:
            # 正常输出
            return ' '.join(tokens)
    
    def transcribe(self, audio: torch.Tensor, method='greedy', beam_size=5) -> str:
        """转录音频"""
        if method == 'greedy':
            tokens = self.greedy_decode(audio)
        else:
            tokens = self.beam_search(audio, beam_size)
        
        return self.decode_tokens(tokens)


def compute_wer(ref: str, hyp: str) -> float:
    """
    WER - 如果都是连续字符（无空格），改为计算编辑距离比率
    """
    ref = ref.strip().lower()
    hyp = hyp.strip().lower()
    
    # 如果 hypothesis 没有空格（字符级输出后处理结果），
    # 也去掉 reference 的空格
    if ' ' not in hyp:
        ref = ref.replace(' ', '')
        hyp = hyp.replace(' ', '')
    else:
        # 标准化空格
        ref = ' '.join(ref.split())
        hyp = ' '.join(hyp.split())
    
    ref_words = ref.split() if ' ' in ref else list(ref)
    hyp_words = hyp.split() if ' ' in hyp else list(hyp)
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    # 编辑距离
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    
    return d[-1][-1] / len(ref_words)


def compute_cer(ref: str, hyp: str) -> float:
    """计算 Character Error Rate"""
    ref = ref.replace(' ', '').strip().lower()
    hyp = hyp.replace(' ', '').strip().lower()
    
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


def evaluate(inference, dataloader, method='greedy', beam_size=5, save_results=None):
    """评估模型"""
    total_wer = 0
    total_cer = 0
    num_samples = 0
    results = []
    
    for batch in tqdm(dataloader, desc=f"Evaluating ({method})"):
        audio_features = batch['audio_features'].to(inference.device)
        audio_lengths = batch['audio_lengths'].to(inference.device)
        references = batch['texts']
        
        for i in range(audio_features.size(0)):
            ref = references[i].strip()
            if not ref:
                continue
            
            audio = audio_features[i:i+1, :audio_lengths[i]]
            hyp = inference.transcribe(audio, method=method, beam_size=beam_size)
            
            wer = compute_wer(ref, hyp)
            cer = compute_cer(ref, hyp)
            
            total_wer += wer
            total_cer += cer
            num_samples += 1
            
            results.append({'ref': ref, 'hyp': hyp, 'wer': wer, 'cer': cer})
    
    metrics = {
        'wer': total_wer / num_samples if num_samples > 0 else 0,
        'cer': total_cer / num_samples if num_samples > 0 else 0,
        'num_samples': num_samples
    }
    
    if save_results:
        with open(save_results, 'w', encoding='utf-8') as f:
            f.write(f"WER: {metrics['wer']:.4f}\n")
            f.write(f"CER: {metrics['cer']:.4f}\n")
            f.write(f"Samples: {num_samples}\n\n")
            f.write("=" * 80 + "\n\n")
            
            for i, r in enumerate(results[:100]):
                f.write(f"Sample {i+1}:\n")
                f.write(f"  REF: {r['ref']}\n")
                f.write(f"  HYP: {r['hyp']}\n")
                f.write(f"  WER: {r['wer']:.4f}, CER: {r['cer']:.4f}\n\n")
        
        print(f"Results saved to {save_results}")
    
    return metrics


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = Tokenizer.from_file(args.tokenizer_file)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id('[PAD]')
    
    model = TransformerASR(
        n_mels=args.n_mels,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        vocab_size=vocab_size,
        pad_token_id=pad_id,
        use_cnn_subsampling=True,
        use_spec_augment=False
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Model loaded from {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    inference = ASRInference(model, tokenizer, device, max_len=args.max_len)
    
    if args.mode == 'interactive':
        print("\n=== Interactive Mode ===")
        print("Post-processing: Merge consecutive single characters")
        
        while True:
            path = input("\nAudio file (or 'quit'): ").strip()
            if path.lower() == 'quit':
                break
            
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
            
            sample = torch.load(path, weights_only=False)
            audio = sample['audio_features'].unsqueeze(0).to(device)
            
            text = inference.transcribe(audio, method=args.method, beam_size=args.beam_size)
            print(f"\nTranscription: {text}")
            
            if 'text' in sample:
                ref = sample['text']
                print(f"Reference:     {ref}")
                
                wer = compute_wer(ref, text)
                cer = compute_cer(ref, text)
                
                print(f"\nMetrics:")
                print(f"  WER: {wer:.4f}")
                print(f"  CER: {cer:.4f}")
    
    else:  # evaluate
        def load_metas(file):
            with open(file, 'r') as f:
                return [line.strip().split()[0] for line in f if line.strip()]
        
        metas = load_metas(os.path.join(args.metadata_dir, f'{args.split}.txt'))
        
        from dataset import LRS2Dataset, collate_fn
        from torch.utils.data import DataLoader
        from functools import partial
        
        dataset = LRS2Dataset(metas, args.dataset_dir, tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=partial(collate_fn, pad_id=pad_id)
        )
        
        print(f"\nEvaluating on {args.split} set ({len(dataset)} samples)")
        print(f"Method: {args.method}" + (f" (beam_size={args.beam_size})" if args.method == 'beam_search' else ""))
        
        metrics = evaluate(
            inference, loader,
            method=args.method,
            beam_size=args.beam_size,
            save_results=args.save_results
        )
        
        print("\n" + "=" * 60)
        print(f"WER: {metrics['wer']:.4f}")
        print(f"CER: {metrics['cer']:.4f}")
        print(f"Samples: {metrics['num_samples']}")
        print("=" * 60)

 # python inference.py --mode interactive --checkpoint ./checkpoints/checkpoint_99.pt  --split train
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', required=True, choices=['interactive', 'evaluate'])
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    
    parser.add_argument('--dataset_dir', default='./dataset')
    parser.add_argument('--metadata_dir', default='./lrs2')
    parser.add_argument('--tokenizer_file', default='tokenizer.json')
    
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=3072)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--method', default='greedy', choices=['greedy', 'beam_search'])
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=200)
    
    parser.add_argument('--split', default='test', choices=['val', 'test', 'train'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_results', default=None, help='Save results to file')
    
    args = parser.parse_args()
    main(args)
