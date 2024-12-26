# dino_pre_extract.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dino_pre import DINOFeatureExtractor  
from custom_dataset import UnlabeledImageDataset  




# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # DINO模型需要224x224的输入
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 加载数据集
root_dirs = [
    'data/FRONT3D/train/rgb',
    'data/FRONT3D/test/rgb',
    'SSR/data/FRONT3D/val/rgb'
]
dataset = UnlabeledImageDataset(root_dirs, transform=transform)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 初始化 DINO 特征提取器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dino_extractor = DINOFeatureExtractor().to(device)
dino_extractor.eval()

# 提取特征并保存
features = []
indices = []
image_names = []

with torch.no_grad():
    for images, batch_indices, batch_image_names in dataloader:
        images = images.to(device)
        cls_features = dino_extractor(images)
        features.append(cls_features.cpu())
        indices.extend(batch_indices)
        image_names.extend(batch_image_names)

# 将特征和索引保存到磁盘
features = torch.cat(features, dim=0)
torch.save({'features': features, 'indices': indices, 'img_name': image_names}, 'dino_features768.pt')