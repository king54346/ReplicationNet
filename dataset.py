"""
LRS2 Dataset - 精简版
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
from functools import partial


class LRS2Dataset(Dataset):
    """加载预处理的.pt文件，自动添加 BOS/EOS"""
    
    def __init__(self, meta_list: List[str], dataset_dir: str, tokenizer):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        
        # 特殊 token IDs
        self.pad_id = tokenizer.token_to_id('[PAD]')
        self.bos_id = tokenizer.token_to_id('[BOS]')
        self.eos_id = tokenizer.token_to_id('[EOS]')
        
        # 过滤存在的样本
        self.valid_samples = [
            meta for meta in meta_list
            if os.path.exists(os.path.join(dataset_dir, f'{meta}.pt'))
        ]
        
        print(f"Dataset: {len(self.valid_samples)} samples")
        print(f"  PAD={self.pad_id}, BOS={self.bos_id}, EOS={self.eos_id}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        meta = self.valid_samples[idx]
        sample = torch.load(
            os.path.join(self.dataset_dir, f'{meta}.pt'),
            weights_only=False
        )
        
        # 获取 tokens
        tokens = sample['tokens']
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        # 确保有 BOS/EOS
        if tokens[0] != self.bos_id:
            tokens = [self.bos_id] + tokens
        if tokens[-1] != self.eos_id:
            tokens = tokens + [self.eos_id]
        
        return {
            'audio_features': sample['audio_features'],
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'text': sample.get('text', '')
        }


def collate_fn(batch: List[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    """Batch collate"""
    audio_features = [item['audio_features'] for item in batch]
    tokens = [item['tokens'] for item in batch]
    texts = [item['text'] for item in batch]
    
    # 记录原始长度
    audio_lengths = torch.tensor([feat.size(0) for feat in audio_features], dtype=torch.long)
    token_lengths = torch.tensor([tok.size(0) for tok in tokens], dtype=torch.long)
    
    # Padding
    audio_padded = pad_sequence(audio_features, batch_first=True, padding_value=0.0)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=pad_id)
    
    return {
        'audio_features': audio_padded,
        'audio_lengths': audio_lengths,  # ✓ 添加
        'tokens': tokens_padded,
        'token_lengths': token_lengths,  # ✓ 添加
        'texts': texts
    }


def get_dataloaders(
    train_metas: List[str],
    val_metas: List[str],
    test_metas: List[str],
    dataset_dir: str,
    tokenizer,
    batch_size: int = 16,
    num_workers: int = 4
):
    """创建数据加载器"""
    pad_id = tokenizer.token_to_id('[PAD]')
    collate_fn_with_pad = partial(collate_fn, pad_id=pad_id)
    
    # 创建数据集
    train_dataset = LRS2Dataset(train_metas, dataset_dir, tokenizer)
    val_dataset = LRS2Dataset(val_metas, dataset_dir, tokenizer)
    test_dataset = LRS2Dataset(test_metas, dataset_dir, tokenizer)
    
    # 创建加载器
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