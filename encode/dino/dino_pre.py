import torch
from torch import nn
import torch.hub
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 定义DINO的预训练模型类
class DINOFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 从torch.hub加载DINO的预训练ViT模型
        self.dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        
        # # 定义MLP，将768维度转换为256维度
        # self.mlp = nn.Sequential(
        #     nn.Linear(768, 256),
        #     nn.LeakyReLU()
        # )

    def forward(self, images):
        # 提取[CLS] token特征
        cls_features = self.dino_model(images)  # 提取[CLS] token，尺寸为[batch_size, 768]
        # transformed_features = self.mlp(cls_features)  # 输出尺寸为[batch_size, 256]
        
        return cls_features
    

