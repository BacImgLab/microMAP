import os
import json
import random
from itertools import combinations
from collections import defaultdict
import numpy as np

def generate_mix_json_optimized(k, tif_folder, bacteria_dict, output_dir, 
                               split_ratio=(12,2,1), samples_per_combo=2, seed=42):
    """
    生成混合菌类数据集的JSON配置文件（优化版）
    
    参数:
    - k: 需要混合的细菌种类数量
    - tif_folder: 包含TIFF文件的文件夹路径
    - bacteria_dict: 细菌名称到视场数字的映射字典
    - output_dir: 输出JSON文件的目录
    - split_ratio: 训练集:验证集:测试集的比例，默认为2:1:1
    - samples_per_combo: 每种细菌组合生成的样本数量，默认为4
    - seed: 随机种子，确保结果可重现
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 验证参数
    if k < 1 or k > len(bacteria_dict):
        raise ValueError(f"k必须在1到{len(bacteria_dict)}之间")
    
    # 获取所有TIFF文件名
    tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]
    print(f"找到 {len(tif_files)} 个TIFF文件")
    
    # 按视场组织文件
    '''
    pos_files = defaultdict(list)
    for file in tif_files:
        # 解析文件名格式: pos_XX_Y.tif
        parts = file.split('_')
        if len(parts) >= 3 and parts[0] == 'pos':
            try:
                pos_num = int(parts[1])
                pos_files[pos_num].append(file)
            except ValueError:
                continue  # 跳过无法解析的文件
    '''
    pos_files = defaultdict(list)
    for file in tif_files:
        parts = file.split(".")
        if len(parts) >= 2:
            try:
                parts = parts[0].split('_')
                pos_num = int(parts[1])
                print(pos_num)
                pos_files[pos_num].append(file)
            except ValueError:
                continue  # 跳过无法解析的文件

    
    # 创建细菌到文件的映射
    bacteria_files = {}
    file_counts = {}
    
    for bacteria, positions in bacteria_dict.items():
        bacteria_files[bacteria] = []
        for pos in positions:
            if pos in pos_files:
                bacteria_files[bacteria].extend(pos_files[pos])
        
        # 去重并排序，确保结果可重现
        bacteria_files[bacteria] = sorted(list(set(bacteria_files[bacteria])))
        file_counts[bacteria] = len(bacteria_files[bacteria])
        print(f"细菌 {bacteria}: {file_counts[bacteria]} 个文件")
    
    # 检查每个细菌是否有足够的文件
    min_files = min(file_counts.values())
    if min_files < samples_per_combo:
        print(f"警告: 有些细菌的文件数量不足 {samples_per_combo} 个")
    
    # 生成所有可能的k种细菌组合
    bacteria_list = list(bacteria_dict.keys())
    print(bacteria_list)
    bacteria_combinations = list(combinations(bacteria_list, k))
    
    print(f"生成 {len(bacteria_combinations)} 种 {k} 种细菌的组合")
    
    # 初始化结果字典
    result = {
        "mix_bacs": [],  # 细菌组合列表
        "mix_fnames": []  # 对应的文件名组合列表
    }
    
    # 为每种细菌组合生成样本
    for combo in bacteria_combinations:
        # 为当前组合中的每种细菌选择文件
        selected_files_per_bacteria = {}
        
        for bacteria in combo:
            available_files = bacteria_files[bacteria]
            # 随机选择 samples_per_combo 个文件，确保不重复
            print(f"sam = {samples_per_combo}")
            if len(available_files) >= samples_per_combo:
                selected_files = random.sample(available_files, samples_per_combo)
            else:
                # 如果文件不足，重复使用某些文件
                selected_files = available_files
                while len(selected_files) < samples_per_combo:
                    selected_files.append(random.choice(available_files))
                selected_files = selected_files[:samples_per_combo]
            
            selected_files_per_bacteria[bacteria] = selected_files
        
        # 生成 samples_per_combo 个样本
        for i in range(samples_per_combo):
            sample_files = []
            
            # 为组合中的每种细菌选择第i个文件
            for bacteria in combo:
                selected_file = selected_files_per_bacteria[bacteria][i]
                # 移除文件扩展名
                selected_file = os.path.splitext(selected_file)[0]
                #sel
                sample_files.append(selected_file)
            
            # 添加到结果中
            result["mix_bacs"].append(list(combo))
            result["mix_fnames"].append(sample_files)
    
    # 计算划分比例
    total_samples = len(result["mix_fnames"])
    train_ratio, val_ratio, test_ratio = split_ratio
    total_parts = train_ratio + val_ratio + test_ratio
    
    train_count = int(total_samples * train_ratio / total_parts)
    val_count = int(total_samples * val_ratio / total_parts)
    test_count = total_samples - train_count - val_count
    
    print(f"总样本数: {total_samples}")
    print(f"训练集: {train_count}, 验证集: {val_count}, 测试集: {test_count}")
    
    # 随机打乱样本顺序
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    # 创建划分
    splits = {
        "train": {
            "mix_bacs": [result["mix_bacs"][i] for i in indices[:train_count]],
            "mix_fnames": [result["mix_fnames"][i] for i in indices[:train_count]]
        },
        "val": {
            "mix_bacs": [result["mix_bacs"][i] for i in indices[train_count:train_count+val_count]],
            "mix_fnames": [result["mix_fnames"][i] for i in indices[train_count:train_count+val_count]]
        },
        "test": {
            "mix_bacs": [result["mix_bacs"][i] for i in indices[train_count+val_count:]],
            "mix_fnames": [result["mix_fnames"][i] for i in indices[train_count+val_count:]]
        }
    }
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个划分保存JSON文件
    for split_name, split_data in splits.items():
        filename = os.path.join(output_dir, f"mix_{k}_{split_name}.json")
        with open(filename, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"已生成: {filename} (包含 {len(split_data['mix_fnames'])} 个样本)")
    
    # 保存统计信息
    stats = {
        "k": k,
        "total_combinations": len(bacteria_combinations),
        "total_samples": total_samples,
        "split_counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count
        },
        "file_counts": file_counts
    }
    
    with open(os.path.join(output_dir, f"mix_{k}_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    return splits, stats

# 你的细菌字典
bacteria_dict = {
    "A. muciniphila": [0,1,2,3,4],
    "B. animalis": [5,6,7,8,9],
    "B. fragilis": [10,11,12,13,14],
    "B. pseudococcoide": [15,16,17,18,19],
    "C. comes": [20,21,22,23,24],
    "C. innocuum": [25,26,27,28,29],
    "E. clostidioformis": [30,31,32,33,34],
    "E. coli": [35,36,37,38,39],
    "E. faecium": [40,41,42,43,44],
    "F. plautii": [45,46,47,48,49],
    "LGG": [50,51,52,53,54]
}
'''
bacteria_dict = {
    "A. muciniphila": [95,96,97,98,99],
    "B. animalis": [66,67,68,69,71],
    "B. fragilis": [111,112,113,114,115],
    "B. Pseudococcoide": [116,117,118,120,121],
    "C. comes": [106,107,108,109,110],
    "C. innocuum": [100,101,102,103,104],
    "E. clostridioformis": [87,88,89,90,91],
    "E. coli": [81,82,83,84,86],
    "E. faecium": [73,74,75,76,77],
    "F. plautii": [131,132,133,134,135],
    "LGG": [124,125,126,129,130]
}
'''

# 使用示例
if __name__ == "__main__":
    # 参数设置
    k = 10  # 混合的细菌种类数量
    tif_folder = "/CX/neuro_tracking/ruifanghai/global_view_20250727_new/im_crop"  # 包含TIFF文件的文件夹
    output_dir = "/CX/neuro_tracking/ruifanghai/global_view_20250727_new/"  # 输出JSON文件的目录
    
    # 生成配置文件
    splits, stats = generate_mix_json_optimized(k, tif_folder, bacteria_dict, output_dir)
    
    # 显示详细统计信息
    print("\n=== 详细统计信息 ===")
    print(f"细菌种类数: {len(bacteria_dict)}")
    print(f"混合种类数 (k): {k}")
    print(f"细菌组合数: {stats['total_combinations']}")
    print(f"总样本数: {stats['total_samples']}")
    print(f"划分比例: {stats['split_counts']}")
    
    # 显示每个细菌的文件数量
    print("\n每个细菌的文件数量:")
    for bacteria, count in stats['file_counts'].items():
        print(f"  {bacteria}: {count} 个文件")
    
    # 显示每个划分的样本示例
    for split_name, split_data in splits.items():
        print(f"\n{split_name} 集前3个样本示例:")
        for i in range(min(3, len(split_data['mix_fnames']))):
            print(f"  样本 {i+1}:")
            print(f"    细菌组合: {split_data['mix_bacs'][i]}")
            print(f"    文件: {split_data['mix_fnames'][i]}")