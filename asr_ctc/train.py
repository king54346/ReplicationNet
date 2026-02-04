"""
CTC DDP Training Script - 8 GPU
"""
import os
import argparse
import math
from pathlib import Path
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tqdm import tqdm

from model import CTCASR
from dataset import LRS2Dataset, collate_fn
from functools import partial


class DDPCTCTrainer:
    def __init__(self, rank, world_size, model, train_loader, val_loader, 
                 optimizer, scheduler, config, backend='nccl'):
        self.rank = rank
        self.world_size = world_size
        self.is_master = (rank == 0)
        self.backend = backend
        
        # 设置device
        if backend == 'nccl':
            self.device = torch.device(f'cuda:{rank}')
        else:  # gloo
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        
        # 损失函数
        self.criterion = nn.CTCLoss(
            blank=config['blank_id'],
            reduction='mean',
            zero_infinity=True
        )
        
        # 只在主进程创建TensorBoard
        if self.is_master:
            self.writer = SummaryWriter(config['log_dir'])
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        self.train_loader.sampler.set_epoch(self.current_epoch)
        
        total_loss = 0
        num_batches = 0
        
        # 只在主进程显示进度条
        iterator = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}") if self.is_master else self.train_loader
        
        for batch in iterator:
            audio = batch['audio_features'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            token_lengths = batch['token_lengths'].to(self.device)
            
            # 前向
            log_probs, output_lengths = self.model(audio, audio_lengths)
            
            # CTC loss需要 (T, B, C) 格式
            log_probs_ctc = log_probs.transpose(0, 1)  # (T', B, vocab_size)
            
            # CTC loss
            loss = self.criterion(
                log_probs_ctc,
                tokens,
                output_lengths,
                token_lengths
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
        
        # 计算平均loss
        avg_loss = total_loss / num_batches
        
        # 跨GPU同步loss
        if dist.is_initialized():
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        iterator = tqdm(self.val_loader, desc="Validating", leave=False) if self.is_master else self.val_loader
        
        for batch in iterator:
            audio = batch['audio_features'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            token_lengths = batch['token_lengths'].to(self.device)
            
            log_probs, output_lengths = self.model(audio, audio_lengths)
            log_probs_ctc = log_probs.transpose(0, 1)
            
            loss = self.criterion(
                log_probs_ctc,
                tokens,
                output_lengths,
                token_lengths
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        # 平均loss
        avg_loss = total_loss / num_batches
        
        # 跨GPU同步
        if dist.is_initialized():
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        return {'loss': avg_loss}
    
    def save_checkpoint(self, is_best=False):
        """只在主进程保存"""
        if not self.is_master:
            return
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model': self.model.module.state_dict(),  # 注意: DDP包装后用.module
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        path = os.path.join(self.config['checkpoint_dir'], f"checkpoint_{self.current_epoch}.pt")
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved (epoch {self.current_epoch+1})")
    
    def train(self):
        """训练主循环"""
        if self.is_master:
            print(f"Training on {self.world_size} GPUs with {self.backend} backend")
            print(f"Train: {len(self.train_loader.dataset)}, Val: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 只在主进程打印和记录
            if self.is_master:
                print(f"Epoch {epoch+1}: Train={train_metrics['loss']:.4f}, Val={val_metrics['loss']:.4f}")
                
                # TensorBoard
                self.writer.add_scalar('train/loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                
                # 保存
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                
                self.save_checkpoint(is_best)
                
                # 定期保存
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(is_best=False)
            
            # 同步所有进程
            try:
                if dist.is_initialized():
                    dist.barrier()
            except Exception as e:
                if self.is_master:
                    print(f"Barrier failed: {e}")
        
        if self.is_master:
            self.writer.close()
            print(f"\n✓ Training completed!")
            print(f"Best validation loss: {self.best_val_loss:.4f}")


def get_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """Warmup + Cosine scheduler"""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_distributed_dataloaders(train_dataset, val_dataset, batch_size, num_workers, pad_id):
    """创建分布式数据加载器"""
    
    # DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        drop_last=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        shuffle=False
    )
    
    # Collate function
    collate_fn_with_pad = partial(collate_fn, pad_id=pad_id)
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_with_pad,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_with_pad,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main_worker(rank, world_size, args):
    """每个进程的主函数"""
    
    # 初始化进程组
    try:
        if args.backend == 'nccl':
            # NCCL: 每个进程绑定到一个GPU
            torch.cuda.set_device(rank)
            device = torch.device(f'cuda:{rank}')
        else:
            # Gloo: 可以不严格绑定
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 初始化进程组
        timeout_seconds = 1800  # 30分钟超时
        dist.init_process_group(
            backend=args.backend,
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=timeout_seconds)
        )
        
    except Exception as e:
        print(f"Process {rank} init failed: {e}")
        raise
    
    # 只在主进程打印
    is_master = (rank == 0)
    
    if is_master:
        print("=" * 70)
        print(f"DDP CTC Training on {world_size} GPUs")
        print(f"Backend: {args.backend}")
        print("=" * 70)
    
    # Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_file)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id('[PAD]')
    blank_id = tokenizer.token_to_id('[BLANK]')
    
    if blank_id != 0:
        raise ValueError(f"BLANK must be ID 0, got {blank_id}")
    
    if is_master:
        print(f"\nTokenizer: {args.tokenizer_file}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  BLANK: {blank_id}, PAD: {pad_id}")
    
    # 加载元数据
    def load_metas(file):
        with open(file, 'r') as f:
            return [line.strip().split()[0] for line in f if line.strip()]
    
    train_metas = load_metas(os.path.join(args.metadata_dir, 'train.txt'))
    val_metas = load_metas(os.path.join(args.metadata_dir, 'val.txt'))
    
    # 创建数据集
    train_dataset = LRS2Dataset(train_metas, args.dataset_dir, tokenizer)
    val_dataset = LRS2Dataset(val_metas, args.dataset_dir, tokenizer)
    
    # 创建分布式数据加载器
    train_loader, val_loader = get_distributed_dataloaders(
        train_dataset, val_dataset,
        args.batch_size, args.num_workers, pad_id
    )
    
    # 创建模型
    model = CTCASR(
        n_mels=args.n_mels,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        vocab_size=vocab_size,
        blank_id=blank_id,
        use_cnn_subsampling=True,
        use_spec_augment=args.use_spec_augment
    ).to(device)
    
    if is_master:
        print(f"\nModel:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Encoder layers: {args.num_encoder_layers}")
        print(f"  d_model: {args.d_model}")
    
    # DDP包装
    if args.backend == 'nccl':
        model = DDP(model, device_ids=[rank], output_device=rank)
    else:
        model = DDP(model)
    
    # 优化器
    base_lr = args.lr
    if args.scale_lr:
        # 线性缩放学习率
        scaled_lr = base_lr * world_size
        if is_master:
            print(f"\nLearning rate scaled: {base_lr} → {scaled_lr}")
        lr = scaled_lr
    else:
        lr = base_lr
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay
    )
    
    # 调度器
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_warmup_scheduler(optimizer, warmup_steps, total_steps)
    
    if is_master:
        print(f"\nTraining config:")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Global batch size: {args.batch_size * world_size}")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Learning rate: {lr:.2e}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Warmup steps: {warmup_steps}")
    
    # 配置
    config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'pad_id': pad_id,
        'blank_id': blank_id,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'max_grad_norm': args.max_grad_norm
    }
    
    # 创建目录（只在主进程）
    if is_master:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # 同步所有进程
    if dist.is_initialized():
        dist.barrier()
    
    # 训练
    trainer = DDPCTCTrainer(
        rank, world_size, model, train_loader, val_loader,
        optimizer, scheduler, config, args.backend
    )
    trainer.train()
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()


def main(args):
    """主函数"""
    # 设置环境变量
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    # 检查GPU数量
    if not torch.cuda.is_available():
        print("No CUDA available!")
        return
    
    world_size = torch.cuda.device_count()
    
    if world_size < args.num_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs but only {world_size} available")
        world_size = args.num_gpus
    
    print(f"Starting DDP training on {world_size} GPUs...")
    print(f"Backend: {args.backend}")
    
    # 启动多进程
    mp.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CTC DDP Training')
    
    # 数据
    parser.add_argument('--dataset_dir', default='./dataset')
    parser.add_argument('--metadata_dir', default='./lrs2')
    parser.add_argument('--tokenizer_file', default='tokenizer_ctc.json')
    
    # 模型
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=12)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # 训练
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_spec_augment', action='store_true',
                       help='Enable SpecAugment')
    parser.add_argument('--scale_lr', action='store_true',
                       help='Scale learning rate by number of GPUs')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Gradient clipping norm')
    
    # DDP
    parser.add_argument('--backend', type=str, default='nccl',
                       choices=['nccl', 'gloo'],
                       help='DDP backend (nccl for GPUs, gloo for CPU/compatibility)')
    parser.add_argument('--num_gpus', type=int, default=8,
                       help='Number of GPUs to use')
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='12355')
    
    # 输出
    parser.add_argument('--checkpoint_dir', default='./checkpoints_ctc')
    parser.add_argument('--log_dir', default='./logs_ctc')
    
    args = parser.parse_args()
    
    main(args)