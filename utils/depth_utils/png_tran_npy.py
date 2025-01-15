import os
import numpy as np
from PIL import Image

# 输入和输出文件夹路径
input_folder = 'depth_anything_png'  # 原始的PNG图像所在文件夹
output_folder = 'depth_anything'  # 目标npy文件所在文件夹

# 如果目标文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith("_depth.png"):
        # 读取PNG深度图像
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert('L')  # 转换为灰度图像
        depth_array = np.array(image)

        # 生成输出npy文件名并保存
        npy_filename = filename.replace('_depth.png', '_pred.npy')
        npy_path = os.path.join(output_folder, npy_filename)

        # 将深度图像转换为npy格式并保存
        np.save(npy_path, depth_array)

        print(f"Converted {filename} to {npy_filename}")

print("All files have been successfully converted!")
