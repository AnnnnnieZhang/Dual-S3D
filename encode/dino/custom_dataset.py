# custom_dataset.py
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# 自定义数据集类
class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.transform = transform
        self.image_paths = []
        for root_dir in root_dirs:
            for img_name in os.listdir(root_dir):
                if os.path.isfile(os.path.join(root_dir, img_name)):
                    self.image_paths.append(os.path.join(root_dir, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, idx, img_name
