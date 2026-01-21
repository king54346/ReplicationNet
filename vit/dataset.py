from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import PILToTensor, Compose
import torchvision


# 手写数字
class MNISTDataset:
    """MNIST 数据集包装器"""

    def __init__(self, root='/home/user/demo/review/DiffusionTransform/mnist', batch_size=128, num_workers=4):
        self.batch_size = batch_size
        self.num_workers = num_workers

        # ===== 数据增强 =====
        # 训练集: 随机旋转、平移等
        self.train_transform = transforms.Compose([
            transforms.Resize(224),  # ViT 通常用 224×224
            transforms.RandomRotation(10),  # 随机旋转 ±10度
            transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
        ])

        # 测试集: 不增强
        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # ===== 加载数据 =====
        self.train_dataset = datasets.MNIST(
            root=root,
            train=True,
            transform=self.train_transform,
            download=True
        )

        self.test_dataset = datasets.MNIST(
            root=root,
            train=False,
            transform=self.test_transform,
            download=True
        )

        print(f"训练集大小: {len(self.train_dataset)}")
        print(f"测试集大小: {len(self.test_dataset)}")

    def get_loaders(self):
        """获取 DataLoader"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True  # 丢弃最后不完整的batch
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, test_loader