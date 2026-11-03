import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os.path as osp
import pandas as pd
from PIL import Image
import math

def plot_violin_per_channel(
    data, 
    save_dir, 
    sampled=False, 
    sample_per_channel=10000, 
    skip_zero=False, 
    filename="chs_norm_violin.png"
):
    """
    data: np.ndarray, shape (60, H, W)
    save_dir: str, directory to save the output image
    sampled: bool, whether to sample pixels or use all
    sample_per_channel: int, number of pixels per channel if sampling
    skip_zero: bool, whether to skip pixels with value 0
    filename: str, filename for the output image
    """

    # === Step1: prepare pixels and channels ===
    if sampled:
        samples = []
        channels = []
        for i in range(60):
            pix = data[i].ravel()
            if skip_zero:
                pix = pix[pix != 0]
            if len(pix) == 0:
                continue
            n_sample = min(sample_per_channel, len(pix))
            sampled_pixels = np.random.choice(pix, size=n_sample, replace=False)
            samples.append(sampled_pixels)
            channels.append(np.full(n_sample, i+1))
        pixels = np.concatenate(samples)
        channels = np.concatenate(channels)
    else:
        pixels = data.reshape(60, -1)
        ch = np.arange(1, 61)[:, None]
        channels = np.broadcast_to(ch, pixels.shape)
        pixels = pixels.ravel()
        channels = channels.ravel()
        if skip_zero:
            mask = pixels != 0
            pixels = pixels[mask]
            channels = channels[mask]

    # === Step2: Create dataframe ===
    df = pd.DataFrame({
        'Pixel Value': pixels,
        'Channel': channels.astype(str)  # 注意：要转成str，后面palette需要
    })

    # === Step3: Prepare color palette ===
    cmap = plt.get_cmap('viridis')
    unique_channels = sorted(df['Channel'].unique(), key=lambda x: int(x))
    palette_dict = {ch: cmap(i / 59) for i, ch in enumerate(unique_channels)}

    # === Step4: Plot ===
    plt.figure(figsize=(24, 8))
    sns.violinplot(
        x='Channel', 
        y='Pixel Value', 
        hue='Channel',
        data=df, 
        palette=palette_dict, 
        inner="quartile", 
        cut=0, 
        scale="width",
        legend=False
    )
    
    # 自动设置y轴范围（防止极端值）
    q_low, q_high = df['Pixel Value'].quantile([0.001, 0.999])
    plt.ylim(q_low, q_high)

    # 每12个通道加一条竖线
    for x in range(12, 60, 12):
        plt.axvline(x-0.5, color='gray', linestyle='--', linewidth=1)

    plt.title('Pixel Value Distribution per Channel')
    plt.xlabel('Channel')
    plt.ylabel('Pixel Value')

    # os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_normalized_violin(data, save_dir, sampled=False, sample_per_channel=1000):
    """
    data: numpy array, shape (60, 512, 512)
    save_path: str, path to save the generated violin plot
    """

    assert isinstance(data, np.ndarray), "data must be a numpy array"
    assert data.shape[0] == 60, "data must have 60 channels"
    
    # 不归一化，直接采样
    samples = []
    channels = []
    for i in range(60):
        pixels = data[i].ravel()
        if sampled:
            pixels = np.random.choice(pixels, size=sample_per_channel, replace=False)
        samples.append(pixels)
        channels.extend([i+1] * len(pixels))
    
    samples = np.concatenate(samples)

    # 整理成 DataFrame
    df = pd.DataFrame({
        'Pixel Value': samples,
        'Channel': channels
    })

    # 配置颜色：前30个蓝色系，后30个橙色系
    palette = sns.color_palette("Blues", 30) + sns.color_palette("Oranges", 30)
    palette_dict = {i+1: palette[i] for i in range(60)}

    # 画图
    plt.figure(figsize=(26, 8))
    sns.violinplot(x='Channel', y='Pixel Value', hue='Channel', data=df, palette=palette_dict,
               inner="quartile", cut=0, density_norm="width", legend=False, dodge=False)

    plt.title('Pixel Distribution per Channel (Original Range)', fontsize=16)
    plt.xlabel('Channel', fontsize=14)
    plt.ylabel('Pixel Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 自动设置y轴范围，排除极端值
    ymin = np.percentile(samples, 1)
    ymax = np.percentile(samples, 99)
    plt.ylim(ymin, ymax)

    # 每12个通道画一条竖线
    for xc in range(12, 60, 12):
        plt.axvline(x=xc-0.5, color='gray', linestyle='--', linewidth=1)

    # 保存图片
    save_path = osp.join(save_dir, "chs_norm_violin.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    # print(f"Violin plot saved to {save_path}")

def infer_grid(n):
    """根据总数n自动推断行数和列数，尽量接近正方形"""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

def plot_all_heatmaps(data, output_dir, filename="heatmap_all.png", cmap='viridis'):
    n = data.shape[0]
    rows, cols = infer_grid(n)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))  # 每张小图大一些
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 小图之间留白

    vmin = np.min(data)
    vmax = np.max(data)

    axes = axes.flatten()  # 保证axes是一维的，方便索引

    for idx in range(n):
        ax = axes[idx]
        im = ax.imshow(data[idx], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Channel {idx}", fontsize=8)
        ax.axis('off')

    # 对于多余的子图（如果有的话），关掉
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    # 单独加 colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()  
  
# 保存单张热力图
def save_heatmaps(data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(data.shape[0]):
        plt.figure(figsize=(5, 5))
        sns.heatmap(data[i], cmap='jet', cbar=False) #'viridis
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(save_dir, f'heatmap_{i:02d}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        
def save_prob(data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(data.shape[0]):
        np.save(os.path.join(save_dir, f'prob_{i:02d}.npy'), data[i])
        
# 将所有小图拼接成一张大图
def combine_heatmaps(save_dir, output_path, rows=10, cols=6):
    images = [Image.open(os.path.join(save_dir, f'heatmap_{i:02d}.png')) for i in range(60)]
    w, h = images[0].size
    big_image = Image.new('RGB', (cols * w, rows * h))

    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        big_image.paste(img, (col * w, row * h))

    big_image.save(output_path)


