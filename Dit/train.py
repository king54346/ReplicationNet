# train_ray.py
import ray
from ray import train
from ray.train import ScalingConfig, CheckpointConfig, RunConfig
from ray.train.torch import TorchTrainer
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
from config import *
from dataset import MNIST
from diffusion import diffusion_schedule
from dit import DiT


def train_func(config):
    """Ray分布式训练函数"""
    
    # ===== 1. 获取分布式训练上下文 =====
    rank = train.get_context().get_world_rank()
    local_rank = train.get_context().get_local_rank()
    world_size = train.get_context().get_world_size()
    
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("=" * 60)
        print(f"分布式训练配置:")
        print(f"  总进程数: {world_size}")
        print(f"  Batch Size (per GPU): {config['batch_size']}")
        print(f"  有效 Batch Size: {config['batch_size'] * world_size}")
        print("=" * 60)
    
    # ===== 2. 初始化模型 =====
    model = DiT(
        img_size=28,
        patch_size=4,
        in_channels=1,
        hidden_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=10,
        class_dropout_prob=0.1
    ).to(device)
    
    model = train.torch.prepare_model(model)
    
    # ===== 3. 优化器和学习率调度器 =====
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['learning_rate'] * 0.01
    )
    
    # ===== 4. 从checkpoint恢复 =====
    start_epoch = 0
    start_iter = 0
    checkpoint = train.get_checkpoint()
    
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(
                os.path.join(checkpoint_dir, "checkpoint.pt"),
                map_location=device
            )
            model.module.load_state_dict(checkpoint_dict["model_state_dict"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])
            start_epoch = checkpoint_dict["epoch"] + 1
            start_iter = checkpoint_dict.get("iter", 0)
            
            if rank == 0:
                print(f"✅ 从 Epoch {start_epoch}, Iter {start_iter} 恢复训练")
                print(f"   上次loss: {checkpoint_dict['loss']:.6f}")
    elif rank == 0:
        print("🆕 从头开始训练")
    
    # ===== 5. 损失函数 =====
    loss_fn = nn.MSELoss()
    
    # ===== 6. 数据加载 =====
    dataset = MNIST()
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True
    )
    
    dataloader = train.torch.prepare_data_loader(dataloader)
    
    # ===== 7. 训练参数 =====
    CHECKPOINT_FREQ_EPOCHS = config.get('checkpoint_freq_epochs', 10)
    PRINT_FREQ = config.get('print_freq', 100)
    
    iter_count = start_iter
    best_loss = float('inf')
    
    # 将扩散参数移到GPU
    global alphas, alphas_cumprod, variance
    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    variance = variance.to(device)
    
    # ===== 8. 训练循环 =====
    for epoch in range(start_epoch, config['epochs']):
        sampler.set_epoch(epoch)
        model.train()
        diffusion_schedule.to(device)
        epoch_loss = 0.0
        epoch_samples = 0
        
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            # 数据预处理
            x = (imgs * 2 - 1).to(device)
            t = torch.randint(0, T, (imgs.size(0),), device=device)
            y = labels.to(device)
            
            # 前向加噪
            x_noisy, noise = diffusion_schedule.forward_add_noise(x, t)
            
            # 模型预测
            pred_noise = model(x_noisy, t, y)
            
            # 计算损失
            loss = loss_fn(pred_noise, noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            epoch_loss += loss.item() * imgs.size(0)
            epoch_samples += imgs.size(0)
            iter_count += 1
            
            # 定期打印
            if iter_count % PRINT_FREQ == 0 and rank == 0:
                avg_loss = epoch_loss / epoch_samples
                current_lr = optimizer.param_groups[0]['lr']
                print(f'[Epoch {epoch:3d}] [Iter {iter_count:6d}] '
                      f'Loss: {loss.item():.6f} | '
                      f'Avg: {avg_loss:.6f} | '
                      f'LR: {current_lr:.6f}')
        
        # Epoch结束
        avg_epoch_loss = epoch_loss / epoch_samples
        scheduler.step()
        
        # 判断是否保存
        should_checkpoint_epoch = (epoch + 1) % CHECKPOINT_FREQ_EPOCHS == 0
        is_best = avg_epoch_loss < best_loss
        is_final = (epoch == config['epochs'] - 1)
        
        if should_checkpoint_epoch or is_best or is_final:
            if is_best:
                best_loss = avg_epoch_loss
            
            if rank == 0:
                checkpoint_dict = {
                    "epoch": epoch,
                    "iter": iter_count,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_epoch_loss,
                    "best_loss": best_loss,
                    "is_best": is_best,
                    "config": config,
                }
                
                checkpoint_dir = Path("./tmp_checkpoint")
                checkpoint_dir.mkdir(exist_ok=True)
                torch.save(checkpoint_dict, checkpoint_dir / "checkpoint.pt")
                
                checkpoint = train.Checkpoint.from_directory(str(checkpoint_dir))
                
                train.report(
                    {
                        "loss": avg_epoch_loss,
                        "epoch": epoch,
                        "iter": iter_count,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "is_best": is_best
                    },
                    checkpoint=checkpoint
                )
                
                if is_best:
                    print(f"🌟 新的最佳模型! Loss: {avg_epoch_loss:.6f} (Epoch {epoch})")
                elif is_final:
                    print(f"🏁 训练完成! Final Loss: {avg_epoch_loss:.6f}")
                else:
                    print(f"💾 Checkpoint保存 (Epoch {epoch}, Loss: {avg_epoch_loss:.6f})")
            else:
                train.report({
                    "loss": avg_epoch_loss,
                    "epoch": epoch,
                    "iter": iter_count
                })
        else:
            train.report({
                "loss": avg_epoch_loss,
                "epoch": epoch,
                "iter": iter_count
            })


def main():
    """主函数"""
    
    # ===== 1. 初始化Ray =====
    ray.init(
        num_gpus=8,
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
        include_dashboard=True,
        ignore_reinit_error=True,
    )
    
    print("=" * 60)
    print(f"🚀 Ray Dashboard: http://0.0.0.0:8265")
    print("=" * 60)
    
    # ===== 2. 训练配置 =====
    train_config = {
        "epochs": 800,
        "batch_size": 300,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "checkpoint_freq_epochs": 20,
        "print_freq": 100,
    }
    
    # ===== 3. Checkpoint配置 (Ray 3.0新方式) =====
    storage_path = os.path.abspath("./ray_results")
    print(f"📂 Checkpoint存储路径: {storage_path}")
    
    checkpoint_config = CheckpointConfig(
        num_to_keep=3,
        checkpoint_score_attribute="loss",
        checkpoint_score_order="min",
    )
    run_config = RunConfig(
            name="dit_training",
            storage_path=storage_path,
            checkpoint_config=checkpoint_config,
            
    )
    # ===== 4. 创建Trainer (不使用RunConfig) =====
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_config,
        scaling_config=ScalingConfig(
            num_workers=8,
            use_gpu=True,
            resources_per_worker={
                "CPU": 4,
                "GPU": 1,
            },
        ),
        run_config=run_config,
    )
    
    # ===== 5. 开始训练 =====
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60 + "\n")
    
    result = trainer.fit()
    
    # ===== 6. 训练完成 =====
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"最终指标: {result.metrics}")
    
    # ===== 7. 保存最佳模型 =====
    if result.checkpoint:
        print(f"\n📦 最佳checkpoint路径: {result.checkpoint}")
        
        with result.checkpoint.as_directory() as checkpoint_dir:
            checkpoint_data = torch.load(
                os.path.join(checkpoint_dir, "checkpoint.pt")
            )
            
            # 保存模型权重
            torch.save(
                checkpoint_data["model_state_dict"],
                "best_model.pth"
            )
            
            # 保存完整checkpoint
            torch.save(
                checkpoint_data,
                "best_checkpoint.pth"
            )
            
            print(f"✅ 最佳模型已保存:")
            print(f"   - 模型权重: best_model.pth")
            print(f"   - 完整checkpoint: best_checkpoint.pth")
            print(f"   - Epoch: {checkpoint_data['epoch']}")
            print(f"   - Loss: {checkpoint_data['loss']:.6f}")
            print(f"   - Best Loss: {checkpoint_data['best_loss']:.6f}")
    
    # ===== 8. 关闭Ray =====
    ray.shutdown()


if __name__ == '__main__':
    main()