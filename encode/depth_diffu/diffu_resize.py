import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleResizeCNN(nn.Module):
    def __init__(self):
        super(SimpleResizeCNN, self).__init__()
        # 定义一个卷积层，调整通道数到1，同时减小特征图尺寸
        self.downsample_conv = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)  # 从1通道到1通道，同时尺寸减半
        # 定义一个卷积层，调整通道数到256
        self.conv = nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1)  # 从1通道到256通道
        # 定义一个上采样层，调整尺寸
        self.upsample = nn.Upsample(size=(242, 324), mode='bilinear', align_corners=True)

    def forward(self, x):
        # 添加一个通道维度
        x = x.unsqueeze(1)  # [12, 484, 648] -> [12, 1, 484, 648]
        
        # 先进行降采样
        x = F.relu(self.downsample_conv(x))  # [12, 1, 484, 648] -> [12, 1, 242, 324]
        
        # 应用卷积层和激活函数
        x = F.relu(self.conv(x))  # [12, 1, 242, 324] -> [12, 256, 242, 324]
        
        # 应用上采样层（此步骤已不需要，因为我们已经降采样到目标尺寸）
        # x = self.upsample(x)  # [12, 256, 484, 648] -> [12, 256, 242, 324]
        
        return x