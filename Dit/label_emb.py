import torch.nn as nn


class LabelEmbedding(nn.Module):
    """类别标签编码"""

    def __init__(self, num_classes, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, dim)

    def forward(self, labels):
        """
        labels: [B] 类别标签 (0-9 for MNIST)
        return: [B, dim]
        """
        return self.embedding(labels)
