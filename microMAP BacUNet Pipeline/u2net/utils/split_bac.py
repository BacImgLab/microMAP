import os
import os.path as osp
import json
import random
import re

# 设置随机种子，保证可复现
random.seed(42)

# 原始目录
input_dir = '/CX/neuro_tracking/huanggw/12round11bacteria/view_im_merge/'

data_dir = '/CX/neuro_tracking/huanggw/12round11bacteria/'

# 获取所有文件名（可根据实际过滤特定后缀名）
tif_files = [f.split(".")[0] for f in os.listdir(input_dir) 
             if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(".tif") ]

# 提取数字并排序
def extract_numbers(filename):
    """从 pos_x_y 格式的文件名中提取两个数字"""
    match = re.match(r'pos_(\d+)$', filename)
    if match:
        return (int(match.group(1)))
    return (float('inf'), float('inf'))  # 不符合格式的排最后

# 按 (x, y) 元组排序
all_files = sorted(tif_files, key=lambda f: extract_numbers(f))
#print(sorted_files)


# 打乱顺序

random.shuffle(tif_files)

# 划分比例
n_total = len(tif_files)
n_train = int(n_total * 0.4)
n_val = int(n_total * 0.1)
n_test = n_total - n_train - n_val

# 划分
train_files = all_files[:n_train]
val_files = all_files[n_train:n_train + n_val]
test_files = all_files[:]


# 构建字典
split_dict = {
    'train': train_files,
    'val': val_files,
    'test': test_files,
    #'all': all_files
}

# 输出到 JSON 文件
output_json = 'split_info_cated.json'
with open(osp.join(data_dir, output_json), 'w') as f:
    json.dump(split_dict, f, indent=2)

print(f"划分完成，共 {n_total} 个文件，已保存到 {output_json}")
