'''
Author: Hai-rf 1224483724@qq.com
Date: 2024-11-08 22:10:52
LastEditors: Hai-rf 1224483724@qq.com
LastEditTime: 2024-11-09 16:24:04
Description: preprosess for dataset items
'''
import random 
from typing import Union, List
import torch
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from torchvision import transforms

class Compose(object):
    def __init__(self, transforms) -> None:
        self.transforms = transforms
        
    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
            
        return image, target
    
class ToTensor(object):
    def __init__(self, max16=20000, compress="norm"):
        self.max16 = max16
        self.compress = compress # log1p or norm
    def __call__(self, image, target):
        image = image.astype(np.float32)
        if self.compress == "log1p":
            image = np.log1p(image)
        elif self.compress == "norm":
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (std + 1e-8)
        elif self.compress == "0-1":
            min = image.min()
            max = image.max()
            image = (image - min) / (max - min + 1e-8)
        elif self.compress == "ch0-1":
            min_per_channel = image.min(axis=(1, 2), keepdims=True)  # shape: (60, 1, 1)
            max_per_channel = image.max(axis=(1, 2), keepdims=True)  # shape: (60, 1, 1)
            image = (image - min_per_channel) / (max_per_channel - min_per_channel + 1e-8)
        elif self.compress == "clip":
            img = image.astype(np.float32)

            # 1. 计算分位数
            p1 = np.percentile(img, 1)   # 1%分位数
            p99 = np.percentile(img, 99.5) # 99%分位数

            # 2. clip到p1~p99区间
            img = np.clip(img, p1, p99)

            # 3. 再做归一化到0-1
            img = (img - p1) / (p99 - p1 + 1e-8)

            # 4. 然后做整体z-score标准化
            mean = img.mean()
            std = img.std()
            image = (img - mean) / (std + 1e-8)
        elif "clip_" in self.compress:
            img = image.astype(np.float32)

            # 1. 计算分位数
            up = int(self.compress.split("_")[1])
            p1 = np.percentile(img, 1)   # 1%分位数
            p99 = np.percentile(img, up) # 99%分位数

            # 2. clip到p1~p99区间
            img = np.clip(img, p1, p99)

            # 3. 再做归一化到0-1
            img = (img - p1) / (p99 - p1 + 1e-8)

            # 4. 然后做整体z-score标准化
            mean = img.mean()
            std = img.std()
            image = (img - mean) / (std + 1e-8)
        elif self.compress == "clip0-1":
            img = image.astype(np.float32)

            # 1. 计算分位数
            p1 = np.percentile(img, 1)   # 1%分位数
            p99 = np.percentile(img, 99) # 99%分位数

            # 2. clip到p1~p99区间
            img = np.clip(img, p1, p99)

            # 3. 再做归一化到0-1
            image = (img - p1) / (p99 - p1 + 1e-8)

        # 5. 转成torch tensor
        image = torch.from_numpy(image).float()
        #在这里对target做一个转换，是黄国维补充上去的，因为torch貌似不支持从np.unit16转换到tensor
        #target = torch.from_numpy(target.astype)这是原本的代码
        target = torch.from_numpy(target.astype(np.int32))
        return image, target

class ToTensor_origin(object):
    def __init__(self):
        self.img = 16
    def __call__(self, image, target):
        image = torch.from_numpy(image).float()
        target = torch.from_numpy(target)
        return image, target
    
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5) -> None:
        self.p = p
        
    def __call__(self, image, target):
        x = 0
        if random.random() < self.p:
            #x = 1
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target
    
class RandomVerticalFlip(object):
    def __init__(self, p=0.5) -> None:
        self.p = p
        
    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target
    
class RandomGaussianBlur(object):
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.gaussian_blur(image, kernel_size=5, sigma=(0.1, 1.0))
        return image, target
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, image, target):
        image = F.normalize(image, mean = self.mean, std = self.std)
        return image, target
    
class Crop(object):
    def __init__(self, size:Union[int, List[int]],) -> None:
        self.size = size # [h, w]
    
    def __call__(self, image, target):
        # 随机生成裁剪参数（top, left, height, width）
        i, j, h, w = T.RandomCrop.get_params(image, output_size=self.size)
        # 用相同参数裁剪 image 和 target
        image = F.crop(image, i, j, h, w)
        target = F.crop(target, i, j, h, w)
        return image, target
    
class Cropxy(object):
    def __init__(self, x=0, y=0, slen=512):
        self.x = x
        self.y = y
        self.slen = slen

    def __call__(self, image:torch.Tensor, target:torch.Tensor):
        image = image[:, self.y: self.y+self.slen, self.x: self.x+self.slen]
        if len(target.size()) == 3:
            target = target[:, self.y: self.y+self.slen, self.x: self.x+self.slen]
        elif len(target.size()) == 2:
            target = target[self.y: self.y+self.slen, self.x: self.x+self.slen]
        return image, target
    
class Resize(object):
    def __init__(self, size:Union[int, List[int]], resize_mask:bool = True) -> None:
        self.size = size # [h, w]
        self.resize_mask = resize_mask
    
    def __call__(self, image, target):
        image = F.resize(image, self.size)
        if self.resize_mask:
            target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, target

class RandomScale(object):
    def __init__(self, scale_factor=0.1, p=0.5) -> None:
        self.scale_range = (1-scale_factor, 1+scale_factor)
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            image = F.resize(image, (int(image.size(1)*scale), int(image.size(2)*scale)))
            target = F.resize(target, (int(target.size(1)*scale), int(target.size(2)*scale)), 
                              interpolation=F.InterpolationMode.NEAREST)
        return image, target

class RandomScaleWithResize:
    def __init__(self, scale_prop=0.5, scale_range=(0.8, 1.2), output_size=(512, 512), fill_value=0):
        """
        初始化随机缩放变换并保持图像为原尺寸。
        :param scale_range: 缩放比例范围，例如 (0.8, 1.2)
        :param output_size: 最终输出图像的尺寸 (width, height)
        :param fill_value: 填充颜色，默认为黑色 (0, 0, 0)
        """
        self.p = scale_prop
        self.scale_range = scale_range
        self.output_size = output_size
        self.fill_value = fill_value

    def __call__(self, image, target):
        """
        应用随机缩放并调整尺寸。
        :param img: 输入图像 (PIL Image)
        :return: 调整后的图像
        """
        if random.random() < self.p:
            scale = random.uniform(*self.scale_range)
            h, w = image.shape[-2:]
            new_w, new_h = int(w * scale), int(h * scale)
            
            # 缩放图像
            image = F.resize(image, (new_h, new_w))
            target = F.resize(target, (new_h, new_w), 
                                interpolation=F.InterpolationMode.NEAREST)
            
            # 调整到原始尺寸
            target_w, target_h = self.output_size
            
            if new_w > target_w or new_h > target_h:
                # 中心裁剪
                left = (new_w - target_w) // 2
                top = (new_h - target_h) // 2
                image = image[:, top:top+target_h, left:left+target_w]
                target = target[:, top:top+target_h, left:left+target_w]
            else:
                # 边缘填充
                delta_h = target_h - new_h
                delta_w = target_w - new_w
                pad_top = delta_h // 2
                pad_bottom = delta_h - pad_top
                pad_left = delta_w // 2
                pad_right = delta_w - pad_left

                image = F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), 
                            padding_mode='constant', fill=self.fill_value)
                target = F.pad(target, (pad_left, pad_top, pad_right, pad_bottom), 
                            padding_mode='constant', fill=self.fill_value)
        return image, target

    
class RandomRotate(object):
    def __init__(self, degree=30, p=0.5):
        self.degree = degree
        self.p = p
        
    def __call__(self, image, target):
        if random.random() < self.p:
            angle = random.uniform(-self.degree, self.degree)
            image = F.rotate(image, angle)
            target = F.rotate(target, angle)
        return image, target
        
        

        

