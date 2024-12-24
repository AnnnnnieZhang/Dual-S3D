import os
import json

# 定义路径
rgb_dir = "data/FRONT3D/train/rgb"
total_train_json_dir = "datasets/FRONT3D_base/split/val"
output_split_dir = "datasets/FRONT3D/split/train"

# 确保输出文件夹存在
os.makedirs(output_split_dir, exist_ok=True)

# 获取目标文件夹中所有 RGB 图片的基名
rgb_files = os.listdir(rgb_dir)
rgb_basenames = {os.path.splitext(f)[0] for f in rgb_files}  # 提取基名（不包含扩展名）

# 遍历 split/train 文件夹中的 JSON 文件
for json_file in os.listdir(total_train_json_dir):
    if not json_file.endswith(".json"):
        continue

    input_json_path = os.path.join(total_train_json_dir, json_file)
    output_json_path = os.path.join(output_split_dir, json_file)

    # 打开 JSON 文件并读取内容
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # 初始化用于存储匹配结果的列表
    matched_data = []

    # 遍历 JSON 数据，匹配基名
    for entry in data:
        rgb_path = entry[0]  # 如 "train/rgb/001758_rgb_009002.jpeg"
        rgb_basename = os.path.splitext(os.path.basename(rgb_path))[0]  # 提取基名

        if rgb_basename in rgb_basenames:
            matched_data.append(entry)

    # 如果有匹配数据，写入输出文件
    if matched_data:
        with open(output_json_path, "w") as f:
            json.dump(matched_data, f, indent=4)

print(f"JSON files have been generated in {output_split_dir}")
