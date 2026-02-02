"""
DDP Training Script - 增强版（包含错误处理和多种后端支持）
"""
import os
import argparse
import math
from pathlib import Path
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tqdm import tqdm

from model import TransformerASR
from dataset import get_dataloaders


def setup_ddp(rank, world_size, backend='nccl'):
    """初始化DDP环境 - 增强版"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 根据backend设置不同的环境变量
    if backend == 'nccl':
        # NCCL调试选项
        os.environ['NCCL_DEBUG'] = 'WARN'  # 可选: INFO, WARN, ERROR
        os.environ['NCCL_TIMEOUT'] = '600'  # 10分钟超时
        # 如果遇到问题，可以尝试这些环境变量：
        # os.environ['NCCL_P2P_DISABLE'] = '1'  # 禁用P2P
        # os.environ['NCCL_IB_DISABLE'] = '1'   # 禁用InfiniBand
        # os.environ['NCCL_SHM_DISABLE'] = '1'  # 禁用共享内存
    
    try:
        # 初始化进程组
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=600)  # 使用datetime.timedelta
        )
        
        if backend == 'nccl':
            torch.cuda.set_device(rank)
        
        if rank == 0:
            print(f"✓ DDP initialized with {backend} backend")
        
        return True
        
    except Exception as e:
        print(f"✗ Rank {rank} failed to initialize DDP: {e}")
        return False


def cleanup_ddp():
    """清理DDP环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


class DDPTrainer:
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
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config['pad_id'],
            label_smoothing=config['label_smoothing']
        )
        
        # 只在主进程创建TensorBoard
        if self.is_master:
            self.writer = SummaryWriter(config['log_dir'])
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        
        # 设置sampler的epoch
        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.current_epoch)
        
        total_loss = 0
        total_tokens = 0
        
        # 只在主进程显示进度条
        iterator = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}") if self.is_master else self.train_loader
        
        for batch in iterator:
            try:
                # 数据移到设备
                audio = batch['audio_features'].to(self.device)
                tokens = batch['tokens'].to(self.device)
                
                # Teacher forcing
                tgt_input = tokens[:, :-1]
                tgt_output = tokens[:, 1:]
                
                # 前向传播
                output = self.model(audio, tgt_input)
                
                # 计算损失
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_output.reshape(-1)
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                # 统计
                batch_tokens = (tgt_output != self.config['pad_id']).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                self.global_step += 1
                
            except RuntimeError as e:
                print(f"Rank {self.rank} encountered error in training: {e}")
                continue
        
        # 聚合所有进程的统计数据
        try:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            total_tokens_tensor = torch.tensor(total_tokens, device=self.device)
            
            if dist.is_initialized():
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
            
            avg_loss = total_loss_tensor.item() / total_tokens_tensor.item() if total_tokens_tensor.item() > 0 else 0
        except Exception as e:
            print(f"Rank {self.rank} failed to reduce metrics: {e}")
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        iterator = tqdm(self.val_loader, desc="Validating", leave=False) if self.is_master else self.val_loader
        
        for batch in iterator:
            try:
                audio = batch['audio_features'].to(self.device)
                tokens = batch['tokens'].to(self.device)
                
                tgt_input = tokens[:, :-1]
                tgt_output = tokens[:, 1:]
                
                output = self.model(audio, tgt_input)
                
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_output.reshape(-1)
                )
                
                batch_tokens = (tgt_output != self.config['pad_id']).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                
            except RuntimeError as e:
                print(f"Rank {self.rank} encountered error in validation: {e}")
                continue
        
        # 聚合所有进程的统计数据
        try:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            total_tokens_tensor = torch.tensor(total_tokens, device=self.device)
            
            if dist.is_initialized():
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
            
            avg_loss = total_loss_tensor.item() / total_tokens_tensor.item() if total_tokens_tensor.item() > 0 else 0
        except Exception as e:
            print(f"Rank {self.rank} failed to reduce val metrics: {e}")
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        
        return {'loss': avg_loss}
    
    def save_checkpoint(self, is_best=False):
        """只在主进程保存检查点"""
        if not self.is_master:
            return
        
        try:
            checkpoint = {
                'epoch': self.current_epoch,
                'model': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'scheduler': self.scheduler.state_dict() if self.scheduler else None
            }
            
            path = os.path.join(self.config['checkpoint_dir'], f"checkpoint_{self.current_epoch}.pt")
            torch.save(checkpoint, path)
            
            if is_best:
                best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
                torch.save(checkpoint, best_path)
                print(f"✓ Best model saved (epoch {self.current_epoch})")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            return
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint['model'])
            
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            if self.scheduler and checkpoint.get('scheduler'):
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            if self.is_master:
                print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    
    def train(self):
        if self.is_master:
            print(f"Training on {self.world_size} GPUs with {self.backend} backend")
            print(f"Train: {len(self.train_loader.dataset)}, Val: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
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
            
            # 同步所有进程
            try:
                if dist.is_initialized():
                    dist.barrier()
            except Exception as e:
                print(f"Barrier failed: {e}")
        
        if self.is_master:
            self.writer.close()


def get_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_distributed_dataloaders(train_metas, val_metas, test_metas, dataset_dir, 
                                tokenizer, batch_size, num_workers, rank, world_size):
    """获取分布式数据加载器"""
    from dataset import LRS2Dataset, collate_fn
    from torch.utils.data import DataLoader
    from functools import partial
    
    # 获取pad_id
    pad_id = tokenizer.token_to_id('[PAD]')
    
    # 创建带pad_id的collate_fn
    collate_fn_with_pad = partial(collate_fn, pad_id=pad_id)
    
    # 创建datasets（不使用is_train参数）
    train_dataset = LRS2Dataset(train_metas, dataset_dir, tokenizer)
    val_dataset = LRS2Dataset(val_metas, dataset_dir, tokenizer)
    test_dataset = LRS2Dataset(test_metas, dataset_dir, tokenizer)
    
    # 创建DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    # 创建DataLoader
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_with_pad,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main_worker(rank, world_size, args):
    """每个进程运行的主函数"""
    try:
        # 初始化DDP
        success = setup_ddp(rank, world_size, backend=args.backend)
        if not success:
            print(f"Rank {rank} failed to setup DDP, exiting...")
            return
        
        # 只在主进程创建目录
        if rank == 0:
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        
        # 等待主进程创建完目录
        if dist.is_initialized():
            dist.barrier()
        
        # Tokenizer
        tokenizer = Tokenizer.from_file(args.tokenizer_file)
        vocab_size = tokenizer.get_vocab_size()
        pad_id = tokenizer.token_to_id('[PAD]')
        
        # 加载元数据
        def load_metas(file):
            with open(file, 'r') as f:
                return [line.strip().split()[0] for line in f if line.strip()]
        
        train_metas = load_metas(os.path.join(args.metadata_dir, 'train.txt'))
        val_metas = load_metas(os.path.join(args.metadata_dir, 'val.txt'))
        test_metas = load_metas(os.path.join(args.metadata_dir, 'test.txt'))
        
        # 分布式数据加载器
        train_loader, val_loader, test_loader = get_distributed_dataloaders(
            train_metas, val_metas, test_metas,
            args.dataset_dir, tokenizer,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            rank=rank,
            world_size=world_size
        )
        
        # 创建模型
        if args.backend == 'nccl':
            device = rank
        else:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
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
        
        # 用DDP包装模型
        if args.backend == 'nccl':
            model = DDP(
                model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False
            )
        else:  # gloo
            model = DDP(
                model,
                find_unused_parameters=False
            )
        
        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {total_params:,}")
        
        # 优化器 (学习率根据GPU数量缩放)
        if args.scale_lr:
            base_lr = args.lr * world_size
        else:
            base_lr = args.lr
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, 0.98),
            weight_decay=args.weight_decay
        )
        
        # 调度器
        total_steps = len(train_loader) * args.num_epochs
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_warmup_scheduler(optimizer, warmup_steps, total_steps)
        
        # 配置
        config = {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'pad_id': pad_id,
            'checkpoint_dir': args.checkpoint_dir,
            'log_dir': args.log_dir,
            'max_grad_norm': 1.0,
            'label_smoothing': 0.0
        }
        
        # 训练器
        trainer = DDPTrainer(rank, world_size, model, train_loader, val_loader, 
                            optimizer, scheduler, config, backend=args.backend)
        
        # 如果需要从检查点恢复
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # 开始训练
        trainer.train()
        
    except Exception as e:
        print(f"Rank {rank} encountered fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        cleanup_ddp()


def main(args):
    """主函数"""
    world_size = torch.cuda.device_count()
    
    if world_size < 1:
        print("Warning: No GPUs available! Falling back to single process training.")
        world_size = 1
    
    # 检查后端兼容性
    if args.backend == 'nccl' and not torch.cuda.is_available():
        print("Warning: NCCL requires CUDA. Switching to Gloo backend.")
        args.backend = 'gloo'
    
    print(f"Found {world_size} GPUs, starting DDP training with {args.backend} backend...")
    
    # 使用torch.multiprocessing启动多进程
    try:
        torch.multiprocessing.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 数据
    parser.add_argument('--dataset_dir', default='./dataset')
    parser.add_argument('--metadata_dir', default='./lrs2')
    parser.add_argument('--tokenizer_file', default='tokenizer.json')
    
    # 模型
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=3072)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # 训练
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--scale_lr', action='store_true', 
                       help='Scale learning rate by number of GPUs')
    
    # DDP配置
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'],
                       help='DDP backend: nccl (GPU) or gloo (CPU/GPU compatible)')
    
    # 输出
    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    parser.add_argument('--log_dir', default='./logs')
    
    # 恢复训练
    parser.add_argument('--resume', default=None, type=str)
    
    args = parser.parse_args()
    main(args)