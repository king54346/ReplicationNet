import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm

from dataset import MNISTDataset
from vit import ViT


class Trainer:
    """训练器"""

    def __init__(
            self,
            model,
            train_loader,
            test_loader,
            device='cuda',
            lr=1e-3,
            weight_decay=0.01,
            epochs=50
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs

        # 优化器和学习率调度器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 记录
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.best_acc = 0.0

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0


        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.epochs} [Train]')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100.0 * correct / total

        return avg_loss, avg_acc

    @torch.no_grad()
    def test_epoch(self, epoch):
        """测试一个epoch"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.test_loader, desc=f'Epoch {epoch + 1}/{self.epochs} [Test]')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # 统计
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(self.test_loader)
        avg_acc = 100.0 * correct / total

        return avg_loss, avg_acc

    def train(self):
        """完整训练流程"""
        print("=" * 60)
        print("开始训练 ViT on MNIST")
        print("=" * 60)

        for epoch in range(self.epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # 测试
            test_loss, test_acc = self.test_epoch(epoch)
            self.test_losses.append(test_loss)
            self.test_accs.append(test_acc)

            # 学习率调度
            self.scheduler.step()

            # 打印统计
            print(f"\nEpoch {epoch + 1}/{self.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.save_checkpoint('best_model.pth')
                print(f"  ✓ 新的最佳模型! Acc: {test_acc:.2f}%")

            print("-" * 60)

        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"最佳测试准确率: {self.best_acc:.2f}%")
        print("=" * 60)

    def save_checkpoint(self, filename):
        """保存checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'test_losses': self.test_losses,
            'test_accs': self.test_accs,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        """加载checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_acc = checkpoint['best_acc']
        self.train_losses = checkpoint['train_losses']
        self.train_accs = checkpoint['train_accs']
        self.test_losses = checkpoint['test_losses']
        self.test_accs = checkpoint['test_accs']
        print(f"✓ 加载checkpoint成功! Best Acc: {self.best_acc:.2f}%")

    def plot_history(self, save_path='training_history.png'):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(self.train_losses, label='Train Loss', marker='o')
        axes[0].plot(self.test_losses, label='Test Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss History')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(self.train_accs, label='Train Acc', marker='o')
        axes[1].plot(self.test_accs, label='Test Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy History')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"✓ 训练历史已保存到: {save_path}")
        plt.show()


# ==================== 可视化预测 ====================

@torch.no_grad()
def visualize_predictions(model, test_loader, device='cuda', num_samples=16):
    """可视化预测结果"""
    model.eval()

    # 获取一个batch
    images, labels = next(iter(test_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]

    # 预测
    logits = model(images)
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1).cpu()

    # 可视化
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_samples):
        img = images[i].cpu().squeeze()

        # 反归一化
        img = img * 0.3081 + 0.1307
        img = torch.clamp(img, 0, 1)

        axes[i].imshow(img, cmap='gray')

        pred = preds[i].item()
        true = labels[i].item()
        conf = probs[i, pred].item()

        color = 'green' if pred == true else 'red'
        axes[i].set_title(
            f'Pred: {pred} (True: {true})\nConf: {conf:.2%}',
            color=color,
            fontsize=10
        )
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    print("✓ 预测可视化已保存到: predictions.png")
    plt.show()


# ==================== 主程序 ====================

def main():
    # ===== 配置 =====
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01

    print(f"使用设备: {DEVICE}")

    # ===== 数据 =====
    print("\n加载数据...")
    dataset = MNISTDataset(batch_size=BATCH_SIZE, num_workers=4)
    train_loader, test_loader = dataset.get_loaders()

    # ===== 模型 =====
    print("\n创建模型...")
    model = ViT(
        img_size=224,
        patch_size=16,
        in_channels=1,
        num_classes=10,
        embed_dim=192,  # Tiny 模型
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1
    )

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")

    # ===== 训练 =====
    print("\n初始化训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS
    )

    # 开始训练
    trainer.train()

    # ===== 绘制历史 =====
    trainer.plot_history()

    # ===== 可视化预测 =====
    print("\n生成预测可视化...")
    visualize_predictions(model, test_loader, device=DEVICE)

    print("\n✅ 所有任务完成!")


if __name__ == '__main__':
    main()