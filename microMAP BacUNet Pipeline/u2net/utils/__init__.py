'''
Author: Hai-rf 1224483724@qq.com
Date: 2024-11-11 21:34:00
LastEditTime: 2024-11-12 02:01:31
LastEditors: Hai-rf 1224483724@qq.com
Description: 
'''
from .train_eval import train_one_epoch, evaluate, create_lr_scheduler, get_params_groups
from .train_eval_mix import train_mix_one_epoch, evaluate_mix
from .train_eval_mix_MC import train_mix_MC_one_epoch, evaluate_mix_MC
from .overlay_vis import OverlayViser

import torch
import numpy as np
import matplotlib.pyplot as plt

def tensor_to_numpy(tensor):
    """
    处理 PyTorch 张量：
    - 变成 NumPy 格式
    - squeeze 去除 batch 和通道维度
    - 归一化到 0-1
    """
    tensor = tensor.detach().cpu().squeeze(0).squeeze(0)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)  # 归一化
    return tensor.numpy()

def plot_edge_res(edge_feature_single, edge_feature_sobel, 
                  image_gray, image_sobel, save_path="edge_visualization.png"):
    """
    可视化 4 张图并保存
    """
    # 1. 转换为 NumPy 格式
    img1 = tensor_to_numpy(edge_feature_single)
    img2 = tensor_to_numpy(edge_feature_sobel)
    img3 = tensor_to_numpy(image_gray)
    img4 = tensor_to_numpy(image_sobel)

    # 2. 创建子图
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # 1 行 4 列
    titles = ["Edge Feature", "Edge Feature Sobel", "Image Gray", "Image Sobel"]
    images = [img1, img2, img3, img4]

    # 3. 遍历绘制
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=12)
        ax.axis("off")  # 隐藏坐标轴

    # 4. 调整间距并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi=300 提高清晰度
    print(f"Saved visualization to {save_path}")



