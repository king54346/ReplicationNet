import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class ClassificationInference:
    """
    分类任务推理器
    支持两种分类器类型：
    1. 'discrete': 在码本索引上分类
    2. 'quantized': 在量化向量上分类
    """
    
    def __init__(self, vqvae_model, classifier_model, device, 
                 classifier_type='discrete', checkpoint_path=None):
        self.vqvae = vqvae_model.to(device)
        self.classifier = classifier_model.to(device)
        self.device = device
        self.classifier_type = classifier_type
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.vqvae.eval()
        self.classifier.eval()
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.vqvae.load_state_dict(checkpoint['vqvae_state'])
        self.classifier.load_state_dict(checkpoint['classifier_state'])
        
        # 从checkpoint中读取分类器类型
        if 'classifier_type' in checkpoint:
            self.classifier_type = checkpoint['classifier_type']
        
        print(f"✓ Models loaded from {path}")
        print(f"  Classifier type: {self.classifier_type}")
        if 'metrics' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
            for k, v in checkpoint['metrics'].items():
                print(f"  {k}: {v}")
    
    @torch.no_grad()
    def predict(self, images):
        """
        预测图像类别
        
        Args:
            images: (B, C, H, W) tensor, 范围[0, 1]
        
        Returns:
            predictions: (B,) 预测类别
            probs: (B, num_classes) 类别概率
            x_recon: (B, C, H, W) 重构图像（用于可视化）
            indices: (B, H, W) 码本索引（用于可视化）
        """
        images = images.to(self.device)
        images_normalized = images * 2 - 1
        
        # VQ-VAE处理
        # 调用forward
        x_recon, _, indices = self.vqvae(images_normalized)
        x_recon = (x_recon + 1) / 2  # 反标准化到[0, 1]
        
        # 根据分类器类型选择输入
        if self.classifier_type == 'discrete':
            # 使用码本索引
            classifier_input = indices
        elif self.classifier_type == 'quantized':
            # 使用量化向量
            z_e = self.vqvae.encode(images_normalized)
            z_q, _, _ = self.vqvae.quantize(z_e)
            classifier_input = z_q
        else:
            # 旧方案：使用重构图像（不推荐）
            classifier_input = x_recon
        
        # 分类
        logits = self.classifier(classifier_input)
        probs = F.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        
        return predictions, probs, x_recon, indices
    
    @torch.no_grad()
    def evaluate(self, data_loader):
        """
        评估整个数据集
        
        Returns:
            accuracy: 准确率
            all_preds: 所有预测
            all_labels: 所有标签
        """
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        
        print("Evaluating...")
        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            predictions, _, _, _ = self.predict(images)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100.0 * correct / total
        
        return accuracy, np.array(all_preds), np.array(all_labels)
    
    def plot_confusion_matrix(self, data_loader, save_path=None):
        """绘制混淆矩阵"""
        _, preds, labels = self.evaluate(data_loader)
        
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (Classifier: {self.classifier_type})')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        plt.show()
    
    def print_classification_report(self, data_loader):
        """打印分类报告"""
        _, preds, labels = self.evaluate(data_loader)
        
        print("\n" + "="*60)
        print(f"Classification Report (Classifier: {self.classifier_type})")
        print("="*60)
        print(classification_report(labels, preds, 
                                    target_names=[str(i) for i in range(10)]))
    
    def visualize_predictions(self, data_loader, num_samples=20, save_path=None):
        """可视化预测结果"""
        # 获取数据
        images_list = []
        labels_list = []
        
        for images, labels in data_loader:
            images_list.append(images)
            labels_list.append(labels)
            if len(images_list) * images.size(0) >= num_samples:
                break
        
        images = torch.cat(images_list)[:num_samples]
        labels = torch.cat(labels_list)[:num_samples]
        
        # 预测
        predictions, probs, x_recon, indices = self.predict(images)
        
        # 绘制
        cols = 5
        rows = (num_samples + cols - 1) // cols
        
        # 根据分类器类型决定显示内容
        if self.classifier_type == 'discrete':
            # 显示：原图、重构图、码本索引分布、预测
            fig = plt.figure(figsize=(cols * 2, rows * 8))
            gs = fig.add_gridspec(rows * 4, cols, hspace=0.3, wspace=0.2)
            
            for i in range(num_samples):
                row = i // cols
                col = i % cols
                
                # 原图
                ax0 = fig.add_subplot(gs[row * 4, col])
                ax0.imshow(images[i, 0], cmap='gray')
                ax0.axis('off')
                if col == 0:
                    ax0.set_ylabel('Original', fontsize=10, rotation=0, ha='right', va='center')
                true_label = labels[i].item()
                ax0.set_title(f'Label: {true_label}', fontsize=9)
                
                # 重构图像
                ax1 = fig.add_subplot(gs[row * 4 + 1, col])
                ax1.imshow(x_recon[i, 0].cpu(), cmap='gray')
                ax1.axis('off')
                if col == 0:
                    ax1.set_ylabel('Recon', fontsize=10, rotation=0, ha='right', va='center')
                
                # 码本索引分布
                ax2 = fig.add_subplot(gs[row * 4 + 2, col])
                idx_flat = indices[i].cpu().flatten().numpy()
                ax2.hist(idx_flat, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                ax2.set_xlim(0, 512)
                ax2.tick_params(labelsize=6)
                ax2.set_ylabel('Count', fontsize=8)
                if col == 0:
                    ax2.set_ylabel('Code Dist', fontsize=10, rotation=0, ha='right', va='center')
                
                # 预测结果
                ax3 = fig.add_subplot(gs[row * 4 + 3, col])
                ax3.axis('off')
                pred_label = predictions[i].item()
                confidence = probs[i, pred_label].item() * 100
                color = 'green' if pred_label == true_label else 'red'
                text = f'Pred: {pred_label}\nConf: {confidence:.1f}%'
                ax3.text(0.5, 0.5, text, ha='center', va='center', 
                        fontsize=10, color=color, weight='bold')
        else:
            # 简化版：原图、重构图、预测
            fig, axes = plt.subplots(rows * 3, cols, figsize=(cols * 2, rows * 6))
            axes = axes.flatten()
            
            for i in range(num_samples):
                # 原图
                ax_img = axes[i * 3]
                ax_img.imshow(images[i, 0], cmap='gray')
                ax_img.axis('off')
                true_label = labels[i].item()
                ax_img.set_title(f'Original: {true_label}', fontsize=8)
                
                # 重构图像
                ax_recon = axes[i * 3 + 1]
                ax_recon.imshow(x_recon[i, 0].cpu(), cmap='gray')
                ax_recon.axis('off')
                ax_recon.set_title('Reconstructed', fontsize=8)
                
                # 预测结果
                ax_pred = axes[i * 3 + 2]
                ax_pred.axis('off')
                pred_label = predictions[i].item()
                confidence = probs[i, pred_label].item() * 100
                color = 'green' if pred_label == true_label else 'red'
                text = f'Pred: {pred_label}\nConf: {confidence:.1f}%'
                ax_pred.text(0.5, 0.5, text, ha='center', va='center', 
                            fontsize=10, color=color, weight='bold')
            
            # 隐藏多余的子图
            for i in range(num_samples * 3, len(axes)):
                axes[i].axis('off')
        
        plt.suptitle(f'Predictions (Classifier: {self.classifier_type})', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Predictions saved to {save_path}")
        plt.show()
    
    def visualize_hard_examples(self, data_loader, num_samples=10, save_path=None):
        """可视化最难分类的样本（低置信度）"""
        all_images = []
        all_labels = []
        all_probs = []
        all_preds = []
        all_recons = []
        all_indices = []
        
        # 收集所有样本
        for images, labels in data_loader:
            predictions, probs, x_recon, indices = self.predict(images)
            
            all_images.append(images)
            all_labels.append(labels)
            all_probs.append(probs.cpu())
            all_preds.append(predictions.cpu())
            all_recons.append(x_recon.cpu())
            all_indices.append(indices.cpu())
        
        all_images = torch.cat(all_images)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)
        all_preds = torch.cat(all_preds)
        all_recons = torch.cat(all_recons)
        all_indices = torch.cat(all_indices)
        
        # 找到置信度最低的样本
        max_probs = all_probs.max(dim=1)[0]
        hard_indices = max_probs.argsort()[:num_samples]
        
        # 绘制
        if self.classifier_type == 'discrete':
            fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 2, 8))
            
            for i, idx in enumerate(hard_indices):
                # 原图
                axes[0, i].imshow(all_images[idx, 0], cmap='gray')
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_ylabel('Original', fontsize=10)
                true_label = all_labels[idx].item()
                axes[0, i].set_title(f'Label: {true_label}', fontsize=8)
                
                # 重构图像
                axes[1, i].imshow(all_recons[idx, 0], cmap='gray')
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_ylabel('Recon', fontsize=10)
                
                # 码本索引分布
                idx_flat = all_indices[idx].flatten().numpy()
                axes[2, i].hist(idx_flat, bins=20, color='coral', alpha=0.7)
                axes[2, i].tick_params(labelsize=6)
                if i == 0:
                    axes[2, i].set_ylabel('Codes', fontsize=10)
                
                # 预测
                pred_label = all_preds[idx].item()
                confidence = max_probs[idx].item() * 100
                axes[3, i].axis('off')
                color = 'green' if pred_label == true_label else 'red'
                axes[3, i].text(0.5, 0.5, f'Pred: {pred_label}\n{confidence:.1f}%', 
                               ha='center', va='center', fontsize=10, color=color, weight='bold')
        else:
            fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
            
            for i, idx in enumerate(hard_indices):
                # 原图
                axes[0, i].imshow(all_images[idx, 0], cmap='gray')
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_ylabel('Original', fontsize=10)
                true_label = all_labels[idx].item()
                axes[0, i].set_title(f'Label: {true_label}', fontsize=8)
                
                # 重构图像
                axes[1, i].imshow(all_recons[idx, 0], cmap='gray')
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_ylabel('Reconstructed', fontsize=10)
                
                # 预测
                pred_label = all_preds[idx].item()
                confidence = max_probs[idx].item() * 100
                axes[2, i].axis('off')
                color = 'green' if pred_label == true_label else 'red'
                axes[2, i].text(0.5, 0.5, f'Pred: {pred_label}\n{confidence:.1f}%', 
                               ha='center', va='center', fontsize=10, color=color, weight='bold')
        
        plt.suptitle(f'Hardest Examples (Classifier: {self.classifier_type})', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Hard examples saved to {save_path}")
        plt.show()
    
    def visualize_code_usage(self, data_loader, save_path=None):
        """可视化码本使用情况（仅适用于discrete类型）"""
        if self.classifier_type != 'discrete':
            print("Code usage visualization only available for 'discrete' classifier")
            return
        
        all_indices = []
        
        print("Collecting code usage statistics...")
        for images, _ in data_loader:
            images = images.to(self.device)
            images_normalized = images * 2 - 1
            
            indices = self.vqvae.get_code_indices(images_normalized)
            all_indices.append(indices.cpu().flatten())
        
        all_indices = torch.cat(all_indices).numpy()
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        axes[0].hist(all_indices, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Codebook Index')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Codebook Usage Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # 排序后的使用频率
        unique, counts = np.unique(all_indices, return_counts=True)
        sorted_counts = sorted(counts, reverse=True)
        
        axes[1].bar(range(len(unique)), sorted_counts, color='coral', alpha=0.7)
        axes[1].set_xlabel('Code Rank (sorted by frequency)')
        axes[1].set_ylabel('Usage Count')
        axes[1].set_title('Codebook Usage (Sorted)')
        axes[1].grid(True, alpha=0.3)
        
        # 统计信息
        usage_ratio = len(unique) / 512
        textstr = f'Used codes: {len(unique)}/512 ({usage_ratio*100:.1f}%)\n'
        textstr += f'Most used: {counts.max()} times\n'
        textstr += f'Least used: {counts.min()} times'
        
        axes[1].text(0.98, 0.98, textstr, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Code usage saved to {save_path}")
        plt.show()
        
        print(f"\nCode usage statistics:")
        print(f"  Used codes: {len(unique)}/512 ({usage_ratio*100:.1f}%)")
        print(f"  Most used code: {counts.max()} times")
        print(f"  Least used code: {counts.min()} times")
    
    def process_single_image(self, image_path, save_path=None):
        """处理单张图像"""
        from PIL import Image
        
        # 读取并预处理
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        
        transform = transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0)
        
        # 预测
        prediction, probs, x_recon, indices = self.predict(img_tensor)
        
        # 可视化
        if self.classifier_type == 'discrete':
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            # 原图
            axes[0].imshow(img_tensor[0, 0], cmap='gray')
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            # 重构图像
            axes[1].imshow(x_recon[0, 0].cpu(), cmap='gray')
            axes[1].set_title('Reconstructed')
            axes[1].axis('off')
            
            # 差异图
            diff = torch.abs(img_tensor[0, 0] - x_recon[0, 0].cpu())
            axes[2].imshow(diff, cmap='hot')
            axes[2].set_title('Absolute Difference')
            axes[2].axis('off')
            
            # 码本索引分布
            idx_flat = indices[0].cpu().flatten().numpy()
            axes[3].hist(idx_flat, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            axes[3].set_xlabel('Code Index')
            axes[3].set_ylabel('Count')
            axes[3].set_title('Code Distribution')
            axes[3].grid(True, alpha=0.3)
            
            # 预测概率
            axes[4].bar(range(10), probs[0].cpu().numpy())
            axes[4].set_xlabel('Digit')
            axes[4].set_ylabel('Probability')
            axes[4].set_title(f'Prediction: {prediction.item()}')
            axes[4].grid(True, alpha=0.3)
        else:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # 原图
            axes[0].imshow(img_tensor[0, 0], cmap='gray')
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            # 重构图像
            axes[1].imshow(x_recon[0, 0].cpu(), cmap='gray')
            axes[1].set_title('Reconstructed')
            axes[1].axis('off')
            
            # 差异图
            diff = torch.abs(img_tensor[0, 0] - x_recon[0, 0].cpu())
            axes[2].imshow(diff, cmap='hot')
            axes[2].set_title('Absolute Difference')
            axes[2].axis('off')
            
            # 预测概率
            axes[3].bar(range(10), probs[0].cpu().numpy())
            axes[3].set_xlabel('Digit')
            axes[3].set_ylabel('Probability')
            axes[3].set_title(f'Prediction: {prediction.item()}')
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✓ Saved to {save_path}")
        plt.show()
        
        return prediction.item(), probs[0].cpu().numpy()


def demo_evaluation():
    """演示评估功能"""
    print("="*60)
    print("Demo: Model Evaluation")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    from model import DeepConvVQVAE
    from train import DiscreteCodeClassifier
    
    vqvae = DeepConvVQVAE(
        in_channels=1,
        image_size=28,
        hidden_channels=[32, 64],
        embedding_dim=64,
        num_embeddings=512,
        num_res_blocks=2
    )
    
    classifier = DiscreteCodeClassifier(
        num_embeddings=512,
        embedding_dim=64,
        spatial_size=7,
        num_classes=10
    )
    
    inferencer = ClassificationInference(
        vqvae_model=vqvae,
        classifier_model=classifier,
        device=device,
        classifier_type='discrete',
        checkpoint_path="checkpoints/classifier_epoch_30.pt"
    )
    
    # 加载测试数据
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 评估
    accuracy, _, _ = inferencer.evaluate(test_loader)
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    # 分类报告
    inferencer.print_classification_report(test_loader)
    
    # 混淆矩阵
    inferencer.plot_confusion_matrix(test_loader, save_path="results/confusion_matrix.png")


def demo_visualizations():
    """演示可视化功能"""
    print("\n" + "="*60)
    print("Demo: Visualizations")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    from model import DeepConvVQVAE
    from train import DiscreteCodeClassifier
    
    vqvae = DeepConvVQVAE(
        in_channels=1,
        image_size=28,
        hidden_channels=[32, 64],
        embedding_dim=64,
        num_embeddings=512,
        num_res_blocks=2
    )
    
    classifier = DiscreteCodeClassifier(
        num_embeddings=512,
        embedding_dim=64,
        spatial_size=7,
        num_classes=10
    )
    
    inferencer = ClassificationInference(
        vqvae_model=vqvae,
        classifier_model=classifier,
        device=device,
        classifier_type='discrete',
        checkpoint_path="checkpoints/classifier_epoch_30.pt"
    )
    
    # 加载测试数据
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 1. 预测可视化
    inferencer.visualize_predictions(test_loader, num_samples=20, 
                                    save_path="results/predictions.png")
    
    # 2. 困难样本
    inferencer.visualize_hard_examples(test_loader, num_samples=10,
                                      save_path="results/hard_examples.png")
    
    # 3. 码本使用情况（仅discrete模式）
    inferencer.visualize_code_usage(test_loader, 
                                   save_path="results/code_usage.png")


def main():
    """运行所有演示"""
    Path("results").mkdir(exist_ok=True)
    
    try:
        demo_evaluation()
    except Exception as e:
        print(f"Evaluation demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        demo_visualizations()
    except Exception as e:
        print(f"Visualization demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("All demos completed!")
    print("="*60)


if __name__ == "__main__":
    main()