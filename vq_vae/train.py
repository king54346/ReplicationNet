import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class DiscreteCodeClassifier(nn.Module):
    """
    直接在VQ-VAE的离散码本索引上进行分类
    这是更合理的架构！
    
    输入：(B, H, W) 的码本索引
    输出：(B, num_classes) 的分类logits
    """
    def __init__(self, num_embeddings=512, embedding_dim=64, spatial_size=7, num_classes=10):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.spatial_size = spatial_size
        
        # 学习码本索引的嵌入（与VQ-VAE的码本共享或独立学习）
        # 这里选择独立学习，让分类器有更大自由度
        self.code_embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # CNN特征提取
        self.features = nn.Sequential(
            # 输入: (B, embedding_dim, H, W)
            nn.Conv2d(embedding_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # (B, 256, 1, 1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, indices):
        """
        Args:
            indices: (B, H, W) 码本索引
        Returns:
            logits: (B, num_classes)
        """
        # 将索引转换为嵌入向量
        # indices: (B, H, W) -> (B, H, W, embedding_dim)
        x = self.code_embedding(indices)
        
        # 转置为CNN格式: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # CNN特征提取
        x = self.features(x)  # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        
        # 分类
        logits = self.classifier(x)
        return logits


class AlternativeClassifier(nn.Module):
    """
    备选方案：在量化向量上分类（而不是索引）
    使用VQ-VAE的码本向量作为特征
    """
    def __init__(self, embedding_dim=64, spatial_size=7, num_classes=10):
        super().__init__()
        
        # 直接在量化后的特征上操作
        self.features = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, z_q):
        """
        Args:
            z_q: (B, embedding_dim, H, W) 量化后的向量
        Returns:
            logits: (B, num_classes)
        """
        x = self.features(z_q)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


class VQVAEClassificationTrainer:
    """
    改进的两阶段训练
    Stage 1: 训练VQ-VAE (无监督)
    Stage 2: 在离散表示上训练分类器 (有监督)
    """
    def __init__(
        self,
        vqvae_model,
        classifier_model,
        device,
        classifier_type='discrete',  # 'discrete' 或 'quantized'
        vqvae_lr=1e-3,
        classifier_lr=1e-3,
        save_dir="checkpoints"
    ):
        self.vqvae = vqvae_model.to(device)
        self.classifier = classifier_model.to(device)
        self.device = device
        self.classifier_type = classifier_type
        
        self.vqvae_optimizer = torch.optim.Adam(self.vqvae.parameters(), lr=vqvae_lr)
        self.classifier_optimizer = torch.optim.Adam(
            self.classifier.parameters(), 
            lr=classifier_lr,
            weight_decay=1e-4  # 添加正则化
        )
        
        # 学习率调度器
        self.classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.classifier_optimizer, 
            T_max=50
        )
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 记录
        self.vqvae_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_losses = []
        self.val_losses = []
    
    # ==================== Stage 1: Train VQ-VAE ====================
    def train_vqvae_epoch(self, train_loader, epoch):
        """训练VQ-VAE"""
        self.vqvae.train()
        total_loss = 0
        total_recon = 0
        total_vq = 0
        
        pbar = tqdm(train_loader, desc=f"[VQ-VAE] Epoch {epoch}")
        for data, _ in pbar:
            data = data.to(self.device)
            data = data * 2 - 1  # 标准化到[-1, 1]
            
            self.vqvae_optimizer.zero_grad()
            
            x_recon, vq_loss, _ = self.vqvae(data)
            recon_loss = F.mse_loss(x_recon, data)
            loss = recon_loss + vq_loss
            
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.vqvae.parameters(), max_norm=1.0)
            
            self.vqvae_optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'vq': f'{vq_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        self.vqvae_losses.append(avg_loss)
        
        return avg_loss, total_recon / len(train_loader), total_vq / len(train_loader)
    
    @torch.no_grad()
    def validate_vqvae(self, val_loader):
        """验证VQ-VAE"""
        self.vqvae.eval()
        total_loss = 0
        
        for data, _ in val_loader:
            data = data.to(self.device)
            data = data * 2 - 1
            
            x_recon, vq_loss, _ = self.vqvae(data)
            recon_loss = F.mse_loss(x_recon, data)
            loss = recon_loss + vq_loss
            
            total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    # ==================== Stage 2: Train Classifier ====================
    def train_classifier_epoch(self, train_loader, epoch):
        """
        训练分类器
        关键改进：在离散表示上分类，而不是重构图像
        """
        self.vqvae.eval()  # 冻结VQ-VAE
        self.classifier.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"[Classifier] Epoch {epoch}")
        for data, labels in pbar:
            data = data.to(self.device)
            labels = labels.to(self.device)
            data_normalized = data * 2 - 1
            
            # 获取离散表示
            with torch.no_grad():
                if self.classifier_type == 'discrete':
                    # 方案1: 使用码本索引
                    indices = self.vqvae.get_code_indices(data_normalized)
                    classifier_input = indices
                else:
                    # 方案2: 使用量化向量
                    z_e = self.vqvae.encode(data_normalized)
                    z_q, _, _ = self.vqvae.quantize(z_e)
                    classifier_input = z_q
            
            # 分类
            self.classifier_optimizer.zero_grad()
            logits = self.classifier(classifier_input)
            loss = F.cross_entropy(logits, labels)
            
            loss.backward()
            self.classifier_optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100.0 * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def validate_classifier(self, val_loader):
        """验证分类器"""
        self.vqvae.eval()
        self.classifier.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for data, labels in val_loader:
            data = data.to(self.device)
            labels = labels.to(self.device)
            data_normalized = data * 2 - 1
            
            # 获取离散表示
            if self.classifier_type == 'discrete':
                indices = self.vqvae.get_code_indices(data_normalized)
                classifier_input = indices
            else:
                z_e = self.vqvae.encode(data_normalized)
                z_q, _, _ = self.vqvae.quantize(z_e)
                classifier_input = z_q
            
            # 分类
            logits = self.classifier(classifier_input)
            loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = 100.0 * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accs.append(avg_acc)
        
        return avg_loss, avg_acc
    
    # ==================== Save & Load ====================
    def save_checkpoint(self, stage, epoch, metrics):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'vqvae_state': self.vqvae.state_dict(),
            'classifier_state': self.classifier.state_dict(),
            'vqvae_optimizer': self.vqvae_optimizer.state_dict(),
            'classifier_optimizer': self.classifier_optimizer.state_dict(),
            'metrics': metrics,
            'classifier_type': self.classifier_type
        }
        path = self.save_dir / f'{stage}_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.vqvae.load_state_dict(checkpoint['vqvae_state'])
        self.classifier.load_state_dict(checkpoint['classifier_state'])
        print(f"✓ Checkpoint loaded from {path}")
        return checkpoint['epoch']
    
    # ==================== Visualization ====================
    def plot_training_curves(self, save_path=None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # VQ-VAE损失
        if self.vqvae_losses:
            axes[0].plot(self.vqvae_losses, label='VQ-VAE Loss', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('VQ-VAE Training Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # 分类准确率
        if self.train_accs:
            axes[1].plot(self.train_accs, label='Train Acc', linewidth=2)
            axes[1].plot(self.val_accs, label='Val Acc', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title('Classification Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # 分类损失
        if self.train_losses:
            axes[2].plot(self.train_losses, label='Train Loss', linewidth=2)
            axes[2].plot(self.val_losses, label='Val Loss', linewidth=2)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Loss')
            axes[2].set_title('Classification Loss')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✓ Training curves saved to {save_path}")
        plt.show()
    
    @torch.no_grad()
    def visualize_predictions(self, data_loader, num_samples=10, save_path=None):
        """可视化预测结果（包含原图、重构图、码本索引分布）"""
        self.vqvae.eval()
        self.classifier.eval()
        
        # 获取数据
        data, labels = next(iter(data_loader))
        data = data[:num_samples].to(self.device)
        labels = labels[:num_samples]
        data_normalized = data * 2 - 1
        
        # VQ-VAE处理
        x_recon, _, indices = self.vqvae(data_normalized)
        x_recon = (x_recon + 1) / 2
        
        # 分类
        if self.classifier_type == 'discrete':
            classifier_input = indices
        else:
            z_e = self.vqvae.encode(data_normalized)
            z_q, _, _ = self.vqvae.quantize(z_e)
            classifier_input = z_q
            
        logits = self.classifier(classifier_input)
        _, predicted = logits.max(1)
        
        # 绘制
        fig = plt.figure(figsize=(num_samples * 1.5, 8))
        gs = fig.add_gridspec(4, num_samples, hspace=0.3, wspace=0.2)
        
        for i in range(num_samples):
            # 原图
            ax0 = fig.add_subplot(gs[0, i])
            ax0.imshow(data[i, 0].cpu(), cmap='gray')
            ax0.axis('off')
            if i == 0:
                ax0.set_ylabel('Original', fontsize=10)
            ax0.set_title(f'Label: {labels[i].item()}', fontsize=9)
            
            # 重构图像
            ax1 = fig.add_subplot(gs[1, i])
            ax1.imshow(x_recon[i, 0].cpu(), cmap='gray')
            ax1.axis('off')
            if i == 0:
                ax1.set_ylabel('Reconstructed', fontsize=10)
            
            # 码本索引分布
            ax2 = fig.add_subplot(gs[2, i])
            idx_flat = indices[i].cpu().flatten().numpy()
            ax2.hist(idx_flat, bins=30, color='steelblue', alpha=0.7)
            ax2.set_xlim(0, self.vqvae.num_embeddings)
            ax2.tick_params(labelsize=6)
            if i == 0:
                ax2.set_ylabel('Code Hist', fontsize=10)
            
            # 预测结果
            ax3 = fig.add_subplot(gs[3, i])
            ax3.axis('off')
            color = 'green' if predicted[i] == labels[i] else 'red'
            ax3.text(0.5, 0.5, f'Pred: {predicted[i].item()}', 
                    ha='center', va='center', fontsize=12, color=color, weight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Predictions saved to {save_path}")
        plt.show()


def prepare_data(batch_size=128, num_workers=4):
    """准备MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    # 设置
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 准备数据
    print("\n" + "="*60)
    print("Preparing data...")
    print("="*60)
    train_loader, val_loader = prepare_data(batch_size=128)
    
    # ==================== Stage 1: Train VQ-VAE ====================
    print("\n" + "="*60)
    print("STAGE 1: Training VQ-VAE (Unsupervised)")
    print("="*60)
    
    from model import DeepConvVQVAE
    
    vqvae = DeepConvVQVAE(
        in_channels=1,
        image_size=28,
        hidden_channels=[32, 64],
        embedding_dim=32,
        num_embeddings=32,
        num_res_blocks=2,
        commitment_cost=0.25,
        use_ema=True,
        ema_decay=0.99
    )
    
    print(f"VQ-VAE parameters: {sum(p.numel() for p in vqvae.parameters()):,}")
    
    # 选择分类器类型
    classifier_type = 'discrete'  # 'discrete' 或 'quantized'
    
    if classifier_type == 'discrete':
        classifier = DiscreteCodeClassifier(
            num_embeddings=32,
            embedding_dim=64,
            spatial_size=7,
            num_classes=10
        )
    else:
        classifier = AlternativeClassifier(
            embedding_dim=64,
            spatial_size=7,
            num_classes=10
        )
    
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    print(f"Classifier type: {classifier_type}\n")
    
    # 创建训练器
    trainer = VQVAEClassificationTrainer(
        vqvae_model=vqvae,
        classifier_model=classifier,
        device=device,
        classifier_type=classifier_type,
        vqvae_lr=1e-3,
        classifier_lr=1e-3
    )
    
    # 训练VQ-VAE
    vqvae_epochs = 30
    best_vqvae_loss = float('inf')
    
    for epoch in range(1, vqvae_epochs + 1):
        train_loss, recon_loss, vq_loss = trainer.train_vqvae_epoch(train_loader, epoch)
        val_loss = trainer.validate_vqvae(val_loader)
        
        print(f"\nEpoch {epoch}/{vqvae_epochs}")
        print(f"  Train Loss: {train_loss:.4f} (Recon: {recon_loss:.4f}, VQ: {vq_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f}")
        
        usage = vqvae.get_codebook_usage()
        perplexity = vqvae.get_perplexity()
        if usage:
            print(f"  Codebook usage: {usage:.2%}, Perplexity: {perplexity:.2f}")
        
        if val_loss < best_vqvae_loss:
            best_vqvae_loss = val_loss
            trainer.save_checkpoint('vqvae', epoch, {'val_loss': val_loss})
        
        print("-" * 60)
    
    # ==================== Stage 2: Train Classifier ====================
    print("\n" + "="*60)
    print(f"STAGE 2: Training Classifier on {classifier_type.upper()} Representation")
    print("="*60)
    
    classifier_epochs = 30
    best_val_acc = 0
    
    for epoch in range(1, classifier_epochs + 1):
        train_loss, train_acc = trainer.train_classifier_epoch(train_loader, epoch)
        val_loss, val_acc = trainer.validate_classifier(val_loader)
        
        # 更新学习率
        trainer.classifier_scheduler.step()
        current_lr = trainer.classifier_optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch}/{classifier_epochs} (LR: {current_lr:.6f})")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trainer.save_checkpoint('classifier', epoch, {
                'train_acc': train_acc,
                'val_acc': val_acc
            })
            print(f"✓ New best accuracy: {val_acc:.2f}%")
        
        if epoch % 5 == 0:
            trainer.visualize_predictions(
                val_loader,
                num_samples=10,
                save_path=f"checkpoints/predictions_epoch_{epoch}.png"
            )
        
        print("-" * 60)
    
    # ==================== Final Results ====================
    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    print(f"Best VQ-VAE Val Loss: {best_vqvae_loss:.4f}")
    print(f"Best Classifier Val Acc: {best_val_acc:.2f}%")
    
    # 绘制曲线
    trainer.plot_training_curves(save_path="checkpoints/training_curves.png")
    
    # 最终可视化
    trainer.visualize_predictions(
        val_loader,
        num_samples=20,
        save_path="checkpoints/final_predictions.png"
    )


if __name__ == "__main__":
    main()