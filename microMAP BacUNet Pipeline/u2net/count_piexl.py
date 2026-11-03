import torch
from torch.utils.data import DataLoader, RandomSampler
from datasets import MixBacDataset, Mix11Dataset,Bac11Dataset, BacteriaDataset, MixDataset, MixScaleDataset, CPDataset

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import transforms as T

class PresetVal:
    def __init__(self, crop_size=512, compress="None",
                 mean=(0, 0, 0), std=(1, 1, 1), max16=10000):
        self.transforms = T.Compose([
            T.ToTensor(max16,compress),
            T.Cropxy(x=68, y=68, slen = crop_size),
        ])

    def __call__(self, img, msk):
        return self.transforms(img, msk)

def compute_and_visualize_class_pixel_stats(data_loader, num_classes, device, save_path):
    """
    单进程环境下统计每个类别的正样本像素总数占比及整体正样本像素总数占比，
    并将结果可视化为柱状图并保存。

    参数:
        data_loader: 数据加载器，每个batch返回 (images, targets, ...)，
                     target形状为 (B, C, H, W)，C为类别数，值为0/1表示正样本。
        num_classes: 类别总数。
        device: 计算设备（如 'cuda' 或 'cpu'）。
        save_path: 图片保存路径（如 './class_pixel_ratios.png'）。

    返回:
        class_ratios: numpy数组，每个类别的正样本像素占比。
        overall_ratio: float，所有类别正样本像素总数占总像素数的比例。
    """
    # 初始化计数器
    counts = torch.zeros(num_classes, dtype=torch.float64, device=device)
    total_pixels = 0  # 总像素数为 B*C*H*W 的累加
    spatial_total_pixels = 0 

    print("正在统计类别像素数...")
    for batch in tqdm(data_loader):
        # 假设 target 在 batch 的第二个元素，可根据实际情况调整索引
        target = batch[1]
        B, C, H, W = target.shape
        if C != num_classes:
            raise ValueError(f"target 的类别数 {C} 与参数 num_classes {num_classes} 不一致！")

        # 累加总像素（每个类别通道单独计数）
        total_pixels += B * C * H * W
        spatial_total_pixels += B * H * W

        # 对 batch 和空间维度求和，得到每个类别的正样本像素数
        batch_counts = target.sum(dim=(0, 2, 3)).to(device, dtype=torch.float64)
        counts += batch_counts

    # 计算占比
    counts_np = counts.cpu().numpy()
    class_ratios = counts_np / spatial_total_pixels          # 每个类别的正样本像素占比
    overall_ratio = counts_np.sum() / total_pixels   # 所有类别正样本像素总数占比

    # 打印简要信息
    print(f"总像素数（类别×空间位置）: {total_pixels}")
    print(f"各类别正样本像素数: {counts_np}")
    print(f"各类别占比: {class_ratios}")
    print(f"整体正样本像素总数占比: {overall_ratio:.6f}")

    # 可视化
    plt.figure(figsize=(12, 6))
    x = np.arange(num_classes)
    bars = plt.bar(x, class_ratios, color='skyblue', edgecolor='black')
    plt.xlabel('类别索引')
    plt.ylabel('正样本像素占比')
    plt.title('每个类别的正样本像素占比')
    plt.xticks(x)
    plt.ylim(0, max(class_ratios.max(), overall_ratio) * 1.1)

    # 在柱状图上显示数值（保留4位小数）
    for bar, ratio in zip(bars, class_ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{ratio:.4f}', ha='center', va='bottom', fontsize=8)

    # 添加整体正样本占比的参考线及标注
    plt.axhline(y=overall_ratio, color='red', linestyle='--', label=f'整体正样本占比 = {overall_ratio:.4f}')
    plt.legend()

    # 确保保存目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化图表已保存至: {save_path}")
    plt.close()  # 关闭图形，释放内存

    return class_ratios, overall_ratio

# 1. 准备数据集和数据加载器
temp_dataset = Bac11Dataset(root_dir = '/CX/neuro_tracking/ruifanghai/',
                                split = "test",
                                transforms=PresetVal(compress='clip'),
                                #transforms=PresetVal(compress=args.in_compress),
                                tr_num=2,
                                gt_name="cp_sf_gt_crop_old",
                                select=False,
                                dataset='global_20250928_data',
                                )
temp_loader = DataLoader(temp_dataset,
                          batch_size=8,
                          num_workers=4,
                          pin_memory=True,
                          sampler=RandomSampler(temp_dataset),
                          )
#dataset = YourDataset(...)  # 实例化您的数据集
#data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

# 2. 定义参数
num_classes = 50           # 您的类别总数，例如 Cityscapes 有 19 类
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = './outputs/class_pixel_ratios.png'  # 图表保存路径

# 3. 调用函数
class_ratios, overall_ratio = compute_and_visualize_class_pixel_stats(
    data_loader=temp_loader,
    num_classes=num_classes,
    device=device,
    save_path=save_path
)

# 4. 可选：后续使用返回的数值
print("各类别占比：", class_ratios)
print("整体正样本像素占比：", overall_ratio)