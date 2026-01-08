from dataset import MNISTDataset
import matplotlib.pyplot as plt
import torch
from vit import ViT
import torch.nn.functional as F

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

test_loader =MNISTDataset().test_dataset # 数据集

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
    ).to(DEVICE) # 模型
model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])

model.eval()    # 预测模式

'''
对图片分类
'''
image,label=test_loader[66]
print('正确分类:',label)
plt.imshow(image.permute(1,2,0))
plt.show()

logits=model(image.unsqueeze(0).to(DEVICE))
print('预测分类:',logits.argmax(-1).item())