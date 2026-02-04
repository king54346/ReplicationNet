"""
LRS2 Dataset for CTC
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
from functools import partial


class LRS2Dataset(Dataset):
    """
    CTC Dataset
    返回audio_features, tokens, audio_length, token_length
    """
    
    def __init__(self, meta_list: List[str], dataset_dir: str, tokenizer):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        
        # 特殊tokens
        self.blank_id = tokenizer.token_to_id('[BLANK]')
        self.pad_id = tokenizer.token_to_id('[PAD]')
        self.bos_id = tokenizer.token_to_id('[BOS]')
        self.eos_id = tokenizer.token_to_id('[EOS]')
        
        # 验证BLANK=0
        if self.blank_id != 0:
            raise ValueError(f"BLANK must be ID 0 for CTC, got {self.blank_id}")
        
        # 过滤存在的样本
        self.valid_samples = [
            meta for meta in meta_list
            if os.path.exists(os.path.join(dataset_dir, f'{meta}.pt'))
        ]
        
        print(f"Dataset: {len(self.valid_samples)}/{len(meta_list)} samples")
        print(f"  BLANK={self.blank_id}, PAD={self.pad_id}, BOS={self.bos_id}, EOS={self.eos_id}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx) -> Dict:
        meta = self.valid_samples[idx]
        sample = torch.load(
            os.path.join(self.dataset_dir, f'{meta}.pt'),
            weights_only=False
        )
        
        # 音频特征
        audio_features = sample['audio_features']  # (T, 80)
        audio_length = sample.get('audio_length', audio_features.size(0))
        
        # Tokens（已包含BOS/EOS）
        tokens = sample['tokens']
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        # 确保有BOS/EOS
        if tokens[0] != self.bos_id:
            tokens = [self.bos_id] + tokens
        if tokens[-1] != self.eos_id:
            tokens = tokens + [self.eos_id]
        
        # 移除可能的BLANK（tokens不应包含blank）
        tokens = [t for t in tokens if t != self.blank_id]
        token_length = len(tokens)
        
        return {
            'audio_features': audio_features,
            'audio_length': audio_length,
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'token_length': token_length,
            'text': sample.get('text', '')
        }


def collate_fn(batch: List[Dict], pad_id: int) -> Dict:
    """
    Batch collate for CTC
    Padding: audio用0, tokens用pad_id
    """
    audio_features = [item['audio_features'] for item in batch]
    audio_lengths = torch.tensor([item['audio_length'] for item in batch], dtype=torch.long)
    
    tokens = [item['tokens'] for item in batch]
    token_lengths = torch.tensor([item['token_length'] for item in batch], dtype=torch.long)
    
    texts = [item['text'] for item in batch]
    
    # Padding
    audio_padded = pad_sequence(audio_features, batch_first=True, padding_value=0.0)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=pad_id)
    
    return {
        'audio_features': audio_padded,      # (B, T_max, 80)
        'audio_lengths': audio_lengths,      # (B,)
        'tokens': tokens_padded,             # (B, L_max)
        'token_lengths': token_lengths,      # (B,)
        'texts': texts                       # List[str]
    }


def get_dataloaders(
    train_metas: List[str],
    val_metas: List[str],
    test_metas: List[str],
    dataset_dir: str,
    tokenizer,
    batch_size: int = 32,
    num_workers: int = 4
):
    """创建数据加载器"""
    pad_id = tokenizer.token_to_id('[PAD]')
    collate_fn_with_pad = partial(collate_fn, pad_id=pad_id)
    
    # 创建数据集
    train_dataset = LRS2Dataset(train_metas, dataset_dir, tokenizer)
    val_dataset = LRS2Dataset(val_metas, dataset_dir, tokenizer)
    test_dataset = LRS2Dataset(test_metas, dataset_dir, tokenizer)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn_with_pad,
        pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn_with_pad,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn_with_pad,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# 测试
# ============================================================================

def test_dataset():
    """测试Dataset"""
    from tokenizers import Tokenizer
    
    print("Testing CTC Dataset...")
    
    # 加载tokenizer
    tokenizer = Tokenizer.from_file('tokenizer_ctc.json')
    
    # 加载metadata
    def load_metas(file):
        with open(file, 'r') as f:
            return [line.strip().split()[0] for line in f if line.strip()]
    
    train_metas = load_metas('./lrs2/train.txt')[:100]
    
    # 创建dataset
    dataset = LRS2Dataset(train_metas, './dataset', tokenizer)
    
    # 测试单个样本
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Audio: {sample['audio_features'].shape}")
    print(f"  Audio length: {sample['audio_length']}")
    print(f"  Tokens: {sample['tokens'].shape}")
    print(f"  Token length: {sample['token_length']}")
    print(f"  Text: {sample['text'][:50]}...")
    
    # 测试DataLoader
    print(f"\nTesting DataLoader...")
    train_loader, _, _ = get_dataloaders(
        train_metas, train_metas[:10], train_metas[:10],
        './dataset', tokenizer,
        batch_size=4, num_workers=0
    )
    
    batch = next(iter(train_loader))
    print(f"\nBatch:")
    print(f"  Audio: {batch['audio_features'].shape}")
    print(f"  Audio lengths: {batch['audio_lengths']}")
    print(f"  Tokens: {batch['tokens'].shape}")
    print(f"  Token lengths: {batch['token_lengths']}")
    
    # CTC验证
    print(f"\nCTC Validation:")
    audio_lens = batch['audio_lengths'] // 4  # 4x subsampling
    token_lens = batch['token_lengths']
    print(f"  Encoded lengths (÷4): {audio_lens}")
    print(f"  Token lengths: {token_lens}")
    print(f"  All audio≥token? {(audio_lens >= token_lens).all().item()}")
    
    print("\n✓ Test passed!")


if __name__ == '__main__':
    test_dataset()