import os
import os.path as osp
import tifffile as tif
import numpy as np
import random
from PIL import Image
from matplotlib import pyplot as plt
#import cv2
import json
from glob import glob
#import h5py
#from collections import defaultdict

dir = "/h3cstore_nt/ruifanghai/global_view_20250731/affine_0.01_10_4.0/im/"
save_dir = "/h3cstore_nt/ruifanghai/global_view_20250731/im_crop"
if not osp.exists(save_dir):
    os.makedirs(save_dir)
    
slen = 650
im_size = 1300 

crop = [[0, 0, slen, slen],
            [0, slen, slen, im_size],
            [slen, 0, im_size, slen],
            [slen, slen, im_size, im_size]
            ]

# 读取tif文件
'''
main_folders = sorted(
    [f for f in os.listdir(dir) 
     if os.path.isdir(os.path.join(dir, f)) and 
        f.startswith("pos")],
    key=lambda x: int(x.split("_")[-1])  # 提取最后一个下划线后的数字
)

#im_list = glob(osp.join(dir, "*.tif"))
for im_path in main_folders:
    #print(im_path)
    im_stack = []
    channel_idx = 0
    if im_path != "pos_58":
        continue
    #print(os.path.join(dir, im_path))
    print(im_path)
    for f in os.listdir(os.path.join(dir, im_path)):
        tif_name = os.path.join(dir, im_path, f)
        if channel_idx <= 4 and f[-5] != 6:
            im = tif.imread(tif_name)
            im = np.array(im)
            im_stack.append(im)
            channel_idx += 1
    #print(im_stack.shape)
    im_stack = np.concatenate(im_stack, axis=0)
    print(im_stack.shape)
    fname = im_path
    for i in range(4):
        out_im_path = osp.join(save_dir, fname+"_"+str(i)+".tif")
        out_im = im_stack[:,crop[i][0]:crop[i][2], crop[i][1]:crop[i][3]]
        print(out_im.shape)
        tif.imwrite(out_im_path, out_im)
      
'''
#main_file = [f for f in os.listdir(dir)],
print(dir)
tif_files = glob(dir + "*.tif")
print(tif_files)
for im_path in tif_files:
    print(im_path)
    im = tif.imread(im_path)
    im = np.array(im)
    print(im.shape)
    fname = im_path.split("/")[-1]
    fname = fname.split(".")[0]
    fname = fname.split("_")[-1]
    print(fname)
    for i in range(4):
        out_im_path = osp.join(save_dir, "pos_"+fname+"_"+str(i)+".tif")
        out_im = im[:,crop[i][0]:crop[i][2], crop[i][1]:crop[i][3]]
        tif.imwrite(out_im_path, out_im)
