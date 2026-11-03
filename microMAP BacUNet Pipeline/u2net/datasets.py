'''
Author: Hai-rf 1224483724@qq.com
LastEditors: Hai-rf 1224483724@qq.com
Description: 
'''
import os
import os.path as osp
import tifffile as tif
import numpy as np
import random
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import json
from glob import glob
import h5py

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

import transforms as T


class PresetTrain:
    def __init__(self, crop_size=512, flip_prop=0.5, compress="norm",
                 mean=(0, 0, 0), std=(1, 1, 1), max16=10000):
        self.transforms = T.Compose([
            T.ToTensor(max16, compress),
            T.Crop(size=(crop_size, crop_size)),
            T.RandomHorizontalFlip(flip_prop),
            # T.Normalize(mean, std)
        ])
        
    def __call__(self, img, msk):
        return self.transforms(img, msk)
    
class PresetEval:
    def __init__(self, crop_size=512, mean=(0, 0, 0), std=(1, 1, 1), max16=10000):
        self.transforms = T.Compose([
            T.ToTensor(max16),
            T.Crop(size=(crop_size, crop_size)),
            # T.Normalize(mean, std)
        ])

    def __call__(self, img, msk):
        return self.transforms(img, msk)
    
class MixPredDataset(Dataset):
    def __init__(self, root_dir:str, train:bool = True, split='val', transforms=None, 
                 crop_size=512, mod=False, select=True, r2ch=True, im_name="im_16bit",
                 ch_mask=False, ch_cl = 0, fixed_value=150,
                 tr_num=4, bac_class=None, collect=False,
                 cl_ch_mapping="ch_cl_mapping_wobg.npy"):
        assert osp.exists(root_dir), f"{root_dir} does not exist."
        self.root = osp.join(root_dir,)
        self.transforms = transforms
        self.mod = mod
        
        if mod:
            im_name = "im_16bit" 
        else:
            im_name = im_name
        self.img_dir = osp.join(self.root, split, im_name)
        assert osp.exists(self.img_dir), f"{self.img_dir} does not exist."
        self.img_paths = glob(osp.join(self.img_dir, 'Pos_*.tif'))
        self.ch_mask = ch_mask
        self.mapping = np.load(cl_ch_mapping, allow_pickle=True)
        self.ch_cl = ch_cl
        self.fixed_value = fixed_value
        # self.r2ch = r2ch
        
    def __len__(self):
        # return 3
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
    
        img = tif.imread(img_path)
        img = img.astype(np.int32)
        # img = img.transpose(1, 2, 0)
        # msk = Image.open(msk_path)
        # img = np.array(img[...,:60]).astype(np.uint8)
        
        
        msk = np.zeros((1, img.shape[0], img.shape[1]))
        if self.transforms is not None:
            img, msk = self.transforms(img, msk)
            h, w = img.shape[:2]
            # x, y = np.random.randint(256, h - 256), np.random.randint(256, w - 256)
            x,y = 650, 650
            lx, ly = 650, 650
            # img, msk = img[:,x-lx:x+lx,y-ly:y+ly], msk[:,x-lx:x+lx,y-ly:y+ly]
            # msk = msk.to(torch.long)
        # if self.r2ch:
        #     indices = []
        #     for i in range(5):
        #         indices.extend(range(i, 60, 5))  # i:5:end 的索引

        #     # 转换为 Tensor 并重排通道
        #     img = img[indices]
        if self.ch_mask:
            row = self.mapping[self.ch_cl]
            zero_chs = np.where(row == 0)[0]
            img[zero_chs, :, :] = self.fixed_value
        
        fname = osp.basename(img_path).split('.')[0]
        bf_info = {
            "crop_loc":(x, y, lx, ly),
            "fname":fname
        }
        if self.ch_mask:
            bf_info["fname"] =  f"{fname}_bac_{self.ch_cl}"
        return img, msk, bf_info    
    
        

class MixScaleDataset(Dataset):
    def __init__(self, root_dir:str, train:bool = True, split='val', transforms=None, 
                 crop_size=512, mod=True, select=True, mix_num=0, k=3,
                 tr_num=4, gt_name="gt", bac_class=None, collect=False,
                 brightness=0.0, contrast=0.0, ctx=250, cty=250):
        assert osp.exists(root_dir), f"{root_dir} does not exist."
        self.root = osp.join(root_dir,)
        self.transforms = transforms
        self.contrast_range = (1-contrast, 1+contrast)
        self.brightness_range = (1-brightness, 1+brightness)
        self.ctx = ctx
        self.cty = cty
        
        self.mod = mod
        ignore_bacs = ["Fn", "Ci", "Eclo", "Pp", 
                            "AKK", "Cc", "Bf", 
                            "Bp", "LGG", "Ba"
                            ]
        # ignore_bacs = ["Ci", "Fn", "Pp", "LGG", 
        #                "Eclo", "Cc", "AKK",
        #                "Bf", "Bp", "Ba"
        #                ]
        self.ignore_bacs = ignore_bacs[:-tr_num]
        if mod:
            im_name = "im_16bit" 
        else:
            im_name = "im"
        self.crops = {
                "train": [0, 1000, 0, 500],
                "val": [0, 500, 500, 1000],
                "test":[500, 1000, 500, 1000],
                "unseen_uncrop": [0, 1000, 0, 1000],
            }
        self.split = "train" if train else split
        if train:
            self.img_dir = osp.join(self.root, 'seen_uncrop', im_name)
            self.msk_dir = osp.join(self.root, 'seen_uncrop', gt_name)
            print(self.msk_dir)
        elif split == 'unseen_uncrop':
            self.img_dir = osp.join(self.root, split, im_name)
            self.msk_dir = osp.join(self.root, split, gt_name)  
        else:
            self.img_dir = osp.join(self.root, 'seen_uncrop', im_name)
            self.msk_dir = osp.join(self.root, 'seen_uncrop', gt_name)
        assert osp.exists(self.img_dir), f"{self.img_dir} does not exist."
        assert osp.exists(self.msk_dir), f"{self.msk_dir} does not exist."
        
        self.bf_dir = osp.join(self.root, 'bf_72')
        
        if bac_class == None:
            self.img_paths = glob(osp.join(self.img_dir, 'Pos_*.tif'))
        else:
            self.img_paths = glob(osp.join(self.img_dir, f'*_{bac_class}.tif'))
        self.msk_paths = glob(osp.join(self.msk_dir, '*.tif'))  
        
        if select:
            selected_imgs = [path for path in self.img_paths
                                if not any(ignore in osp.basename(path) for ignore in self.ignore_bacs)
                                ]
            selected_msks = [path for path in self.msk_paths
                                if not any(ignore in osp.basename(path) for ignore in self.ignore_bacs)
                                ]   
            self.img_paths = selected_imgs
            self.msk_paths = selected_msks
            
        if mix_num == 1:
            new_img = [path for path in self.img_paths if ignore_bacs[-tr_num]  in path]
            new_msk = [path for path in self.msk_paths if ignore_bacs[-tr_num]  in path]
            self.img_paths.extend(new_img * k)
            self.msk_paths.extend(new_msk * k)
        print(len(self.img_paths))    
    def __len__(self):
        # return 3
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        msk_path = osp.join(self.msk_dir, osp.basename(img_path))
        img = tif.imread(img_path)
        msk = tif.imread(msk_path)
          
        
        if self.brightness_range != (1, 1):
            brightness_factor = random.uniform(*self.brightness_range)
            img = img * brightness_factor
        if self.contrast_range != (1, 1):
            contrast_factor = random.uniform(*self.contrast_range)
            mean = np.mean(img)
            img = (img - mean) * contrast_factor + mean
        img = np.clip(img, 0, 65535)
        # msk = Image.open(msk_path)
        img = np.array(img).astype(np.uint8)
        msk = np.array(msk).astype(np.uint8)
        if img.shape[0] != 1000:
            img = img.transpose((1, 2, 0))
        x1, x2, y1, y2 = self.crops[self.split]
        img = img[x1:x2, y1:y2,  :60]
        img = img.transpose((2, 0,1))
        msk = msk[x1:x2, y1:y2,  :]
        msk = msk.transpose((2, 0, 1))
        if self.transforms is not None:
            h, w = self.crops[self.split][1] - self.crops[self.split][0], self.crops[self.split][3] - self.crops[self.split][2]
            # x, y = np.random.randint(250, h - 250), np.random.randint(250, w - 250)
            # x,y = 256,256
            if self.split == 'train':
                x = np.random.randint(250, h - 250)
                y = 250
            if self.split == 'unseen_uncrop':
                x = self.ctx
                y = self.cty
            else:
                x = 250
                y=250
             
            lx, ly = 250, 250
            img, msk = img[:,x-lx:x+lx,y-ly:y+ly], msk[:,x-lx:x+lx,y-ly:y+ly]
            img, msk = self.transforms(img, msk)
            # msk = msk.to(torch.long)
        fname = osp.basename(img_path)
        bf_info = {
            "crop_loc":(x, y, lx, ly),
            "fname":fname
        }
        return img, msk, bf_info

class MixDataset(Dataset):
    def __init__(self, root_dir:str, train:bool = True, split='val', transforms=None, 
                 crop_size=512, mod=True, select=True,
                 tr_num=4, bac_class=None, collect=False,
                 brightness=0.0, contrast=0.0, ctx=None, cty=None):
        assert osp.exists(root_dir), f"{root_dir} does not exist."
        self.root = osp.join(root_dir,)
        self.transforms = transforms
        self.contrast_range = (1-contrast, 1+contrast)
        self.brightness_range = (1-brightness, 1+brightness)
        self.ctx = ctx
        self.cty = cty
        
        self.mod = mod
        ignore_bacs = ["Fn", "Ci", "Eclo", "Pp", 
                            "AKK", "Cc", "Bf", 
                            "Bp", "LGG", "Ba"
                            ]
        self.ignore_bacs = ignore_bacs[:-tr_num]
        if mod:
            im_name = "im_16bit" 
        else:
            im_name = "im"
        self.crops = {
                "train": [0, 1000, 0, 500],
                "val": [0, 500, 500, 1000],
                "test":[500, 1000, 500, 1000],
                "unseen_uncrop": [0, 1000, 0, 1000],
                "seen_uncrop": [0, 1000, 0, 1000],
            }
        self.split = "train" if train else split
        if train:
            self.img_dir = osp.join(self.root, 'train', im_name)
            self.msk_dir = osp.join(self.root, 'train', 'gt_corre')
        else:
            self.img_dir = osp.join(self.root, split, im_name)
            self.msk_dir = osp.join(self.root, split, 'gt_corre')  
        assert osp.exists(self.img_dir), f"{self.img_dir} does not exist."
        assert osp.exists(self.msk_dir), f"{self.msk_dir} does not exist."
        
        self.bf_dir = osp.join(self.root, 'bf_72')
        
        if bac_class == None:
            self.img_paths = glob(osp.join(self.img_dir, 'Pos_*.tif'))
        else:
            self.img_paths = glob(osp.join(self.img_dir, f'*_{bac_class}.tif'))
        self.msk_paths = glob(osp.join(self.msk_dir, '*.tif'))  
        
        if select:
            selected_imgs = [path for path in self.img_paths
                                if not any(ignore in osp.basename(path) for ignore in self.ignore_bacs)
                                ]
            selected_msks = [path for path in self.msk_paths
                                if not any(ignore in osp.basename(path) for ignore in self.ignore_bacs)
                                ]   
            # self.img_paths.sort()
            # self.msk_paths.sort()
            # selected_imgs = []
            # selected_msks = []  
            # for ip, mp in zip(self.img_paths, self.msk_paths):
            #     iname, mname = osp.basename(ip), osp.basename(mp)
            #     assert iname[:-4] == mname[:-4], f"{ip} and {mp} do not match."
            #     bac_name = iname[:-4].split('_')[-1]
            #     if bac_name not in self.ignore_bacs:
            #         selected_imgs.append(ip)
            #         selected_msks.append(mp)

            self.img_paths = selected_imgs
            self.msk_paths = selected_msks
            print(len(self.img_paths))

    def __len__(self):
        # return 3
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        msk_path = osp.join(self.msk_dir, osp.basename(img_path))
        img = tif.imread(img_path)
        msk = tif.imread(msk_path)
        
        # if self.brightness_range != (1, 1):
        #     brightness_factor = random.uniform(*self.brightness_range)
        #     img = img * brightness_factor
        # if self.contrast_range != (1, 1):
        #     contrast_factor = random.uniform(*self.contrast_range)
        #     mean = np.mean(img)
        #     img = (img - mean) * contrast_factor + mean
        # img = np.clip(img, 0, 65535)
        # msk = Image.open(msk_path)
        # img = img.transpose(1, 2, 0)
        img = np.array(img[...,:60]).astype(np.uint8)
        msk = np.array(msk).astype(np.uint8)
        img = img.transpose((2,0,1))
        msk = msk.transpose((2, 0, 1))
        if self.transforms is not None:
            h, w = self.crops[self.split][1] - self.crops[self.split][0], self.crops[self.split][3] - self.crops[self.split][2]
            if self.ctx==None:
                # x, y = np.random.randint(250, h - 250), np.random.randint(250, w - 250)
                x,y = 256,256
                lx, ly = 256, 256
            else:
                x, y = self.ctx, self.cty
                lx, ly = 250, 250
            img, msk = img[:,x-lx:x+lx,y-ly:y+ly], msk[:,x-lx:x+lx,y-ly:y+ly]
            
            img, msk = self.transforms(img, msk)
            # msk = msk.to(torch.long)
        fname = osp.basename(img_path)
        bf_info = {
            "crop_loc":(x, y, lx, ly),
            "fname":fname
        }
        return img, msk, bf_info
    
class Bac11Dataset(Dataset):
    def __init__(self, root_dir:str, dataset="SingleBac11", split='train', 
                 gt_name = "cp_sf_gt_crop",im_name = "im_crop",
                 transforms=None, tr_num=3, select=True, split_json = 'split_info_cate.json'):
        assert osp.exists(root_dir), f"{root_dir} does not exist."
        self.root = osp.join(root_dir, dataset)
        self.transforms = transforms
        self.split = split
        with open(osp.join(self.root, split_json), 'r') as f:
            info = json.load(f)
        
        self.img_dir = osp.join(self.root, im_name)
        self.gt_dir = osp.join(self.root, gt_name)
        if gt_name == "gt_ov_crop":
            self.gt_postfix = ".png"
        else:
            self.gt_postfix = ".tif"
        
        #ignore_bac =["Ba", "LGG", "AKK", "Ecoli", "Ci",
        #             "Fp", "Cc",
        #             "Bf", "Bp", "Eclo", "Ef"]
        #"C. comes",
        #"B. animalis",  "B. Pseudococcoide",
        ignore_bac = ['C. comes', 'B. fragilis', 'B. pseudococcoide', 'E. clostidioformis', 'LGG', 'E. faecium', 'B. animalis', 'A. muciniphila', 'C. innocuum', 'E. coli', 'F. plautii']
        #ignore_bac = ["E. clostridioformis", "A. muciniphila", "B. animalis", "B. Pseudococcoide", "LGG", "F. plautii", "C. comes", "E. faecium", "B. fragilis", "C. innocuum", "E. coli"]
        ignore_bacs = ignore_bac[-tr_num:]
        if tr_num == 0:
            ignore_bacs = []
        #ignore_bac = ["A. muciniphila", "E. faecium", "E. coli", "C. innocuum", "B. fragilis", "C. comes", "B. animalis", 
        #"E. clostridioformis", 
        #"F. plautii", "LGG"]
        #ignore_bacs = ignore_bac[:]
        
        #j加一行print代码用来做测试
        if select:
            #print
            fname_list = []
            for bac, fn_list in info["cate_"+split].items():
                #print("bac = ", bac)
                #print("fn_list = ", fn_list)
                if bac not in ignore_bacs:
                    print(f"here")
                    fname_list.extend(fn_list)
            self.fname_list = fname_list
        else:
            if split == "all":
                self.fname_list = [fname for k in info.keys() for fname in info[k] ]
                # self.fname_list = list(chain.from_iterable(info.values()))
            else:
                self.fname_list = info[split]
                print(f"train or val")
                

    
    def __len__(self):
        return len(self.fname_list)
    
    def __getitem__(self, idx):
        fname = self.fname_list[idx]
        #print(f"fname = {fname}")
        img_path = osp.join(self.img_dir, fname+".tif")
        msk_path = osp.join(self.gt_dir, fname+self.gt_postfix)
        img = tif.imread(img_path)
        assert img is not None, f"{img_path} failed to fetch."
        if self.gt_postfix == ".png":
            msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        else:
            msk = tif.imread(msk_path)
            '''
            max_value = max(msk)
            print(max_value)  # 输出: 9
            '''
        assert msk is not None, f"{msk_path} failed to fetch."
        if self.transforms is not None:
            img, msk = self.transforms(img, msk)
            msk = msk.to(torch.long)
        if self.gt_postfix == ".tif":
            bf_info ={
                "fname":fname,
                "bacn":fname
            }
            return img, msk, bf_info
        else:
            return img, msk, fname
        
class Mix11Dataset(Dataset):
    def __init__(self, root_dir:str, dataset:str, split='train', 
                 gt_name = "cp_sf_gt_crop", im_name = "im_crop",bg_thresh=None,
                 transforms=None, tr_num=11, select=True,
                 mix_num=2, match_hist=False,
                 contrast=0.0, brightness = 0.0, save_imposed_dir=None):
        assert osp.exists(root_dir), f"{root_dir} does not exist."
        self.root = osp.join(root_dir, dataset)
        self.transforms = transforms
        self.split = split
        self.mix_num = mix_num
        with open(osp.join(self.root, f'mix_{mix_num}_{split}.json'), 'r') as f:
            info = json.load(f)
        self.im_size = 650
        self.msk_size = 650
        self.ch_num = 11 # input gt channel number, wo bg.
        self.fluo_chs = 60 # input im channel number
        self.img_dir = osp.join(self.root, im_name)
        self.gt_dir = osp.join(self.root, gt_name)
        
        self.bg_thresh = bg_thresh
        
        self.contrast_range = (1-contrast, 1+contrast)
        self.brightness_range = (1-brightness, 1+brightness)
        
        # ignore_bac =["Ba", "LGG", "AKK", "Ecoli", "Ci",
        #              "Fp", "Cc",
        #              "Bf", "Bp", "Eclo", "Ef"]
        ignore_bac = ["Ci", "Fp", "Ecoli", "LGG", "Ba", "Ef",
                      "AKK", "Bp", "Cc", "Eclo", "Bf"]
        ignore_bacs = ignore_bac[:-tr_num]
        
        if select:
            mix_fnames = []
            for bacs, fns in zip(info["mix_bacs"], info["mix_fnames"]):
                if not any(ignore in bacs for ignore in ignore_bacs):
                    mix_fnames.append(fns)
            self.mix_fnames = mix_fnames
        else:
            #print("here")
            self.mix_fnames = info["mix_fnames"]
        # if im_name == "im_crop_cellpose_merge":
        #     self.mix_fnames = os.listdir(self.img_dir)
        self.im_name = im_name
        self.save_imposed_dir = save_imposed_dir
        if self.save_imposed_dir is not None:
            os.makedirs(osp.join(self.save_imposed_dir, 'images'), exist_ok=True)
            os.makedirs(osp.join(self.save_imposed_dir, 'masks'), exist_ok=True)
        
    def __len__(self):
        return len(self.mix_fnames)
    
    def impose(self, fnames):
        overlay_msk = np.zeros((self.ch_num, self.msk_size, self.msk_size), dtype=np.uint8)
        imposed = np.zeros((self.fluo_chs, self.msk_size, self.msk_size), dtype=np.int32)
        s_part = []
        b_part = []
        p_part = []
        bn_part = []
        mbn_part = []
        part = []
        name = ''
        bac_name = ''
        for fname in fnames:
            #print(fname)
            #print(self.img_dir)
            img_path = osp.join(self.img_dir, fname+".tif")
            msk_path = osp.join(self.gt_dir, fname+".tif")
            img = tif.imread(img_path)
            msk = tif.imread(msk_path)
            # img = np.array(img, dtype=np.int32)
            # # mm = img <=self.bg_thresh
            if self.bg_thresh is not None:
                mm = np.where(img <=self.bg_thresh)
                # print(mm.shape)
                img[mm] = img[mm] // self.mix_num
            imposed += img
            for i in range(self.ch_num):
                overlay_msk[i, :,:] |= msk[i, :, :]
            
            #parts = fname.split("_")
            #s_part.append(parts[-1])
            #parts = parts[:-1]
            #b_part.append(parts[1])
            #p_part.append(parts[3])
            #bn_part.append(parts[4])
            #if len(parts) > 5:
            #    mbn_part.extend(parts[5:])
            #part.append(fname)
            name = name + fname
            bac_name = bac_name + fname
        
        # imposed = imposed // self.mix_num  
        #name = fname
        #bac_name = fname
        return imposed, overlay_msk, name, bac_name
    
    def __getitem__(self, idx):
        fnames = self.mix_fnames[idx]
        # if self.im_name == "im_crop_cellpose_merge":
        #     fnames = [fnames.split(".")[0]]
        img, msk, fname, bn_part = self.impose(fnames)

         # ===== 新增保存功能 =====
        if self.save_imposed_dir is not None:
            # 构造唯一文件名，例如 "000123_fname1_fname2"
            save_name = f"{idx:06d}_{fname}"   # fname 由 impose 拼接得到
            img_path = osp.join(self.save_imposed_dir, 'images', save_name + '.tif')
            msk_path = osp.join(self.save_imposed_dir, 'masks', save_name + '.tif')
            tif.imwrite(img_path, img.astype(np.int32))   # 保持原始类型
            tif.imwrite(msk_path, msk.astype(np.uint8))


        if self.brightness_range != (1, 1):
            brightness_factor = random.uniform(*self.brightness_range)
            img = img * brightness_factor
        if self.contrast_range != (1, 1):
            contrast_factor = random.uniform(*self.contrast_range)
            mean = np.mean(img)
            img = (img - mean) * contrast_factor + mean
        
        if self.transforms is not None:
            img, msk = self.transforms(img, msk)
            msk = msk.to(torch.long)
           
        
        bf_info ={
                "fname":fname,
                "bacn":"_".join(bn_part)
                
        }
        return img, msk, bf_info

class CPDataset(Dataset):
    def __init__(self, root_dir:str, dataset:str, split='train', 
                 gt_name = "gt_crop", im_name = "im_crop_cellpose_merge",
                 bg_thresh=None,
                 transforms=None, tr_num=11, select=True,
                 mix_num=2, match_hist=False,
                 contrast=0.0, brightness = 0.0):
        assert osp.exists(root_dir), f"{root_dir} does not exist."
        self.root = osp.join(root_dir, dataset)
        self.transforms = transforms
        self.split = split
        self.mix_num = mix_num
        with open(osp.join(self.root, f'mix_{mix_num}_{split}.json'), 'r') as f:
            info = json.load(f)
        self.im_size = 650
        self.msk_size = 650
        self.ch_num = 11 # input gt channel number, wo bg.
        self.fluo_chs = 60 # input im channel number
        self.img_dir = osp.join(self.root, im_name)
        self.gt_dir = osp.join(self.root, gt_name)
        
        self.bg_thresh = bg_thresh
        
        self.contrast_range = (1-contrast, 1+contrast)
        self.brightness_range = (1-brightness, 1+brightness)
        
        # ignore_bac =["Ba", "LGG", "AKK", "Ecoli", "Ci",
        #              "Fp", "Cc",
        #              "Bf", "Bp", "Eclo", "Ef"]
        ignore_bac = ["Ci", "Fp", "Ecoli", "LGG", "Ba", "Ef",
                      "AKK", "Bp", "Cc", "Eclo", "Bf"]
        ignore_bacs = ignore_bac[:-tr_num]
        
        if select:
            mix_fnames = []
            for bacs, fns in zip(info["mix_bacs"], info["mix_fnames"]):
                if not any(ignore in bacs for ignore in ignore_bacs):
                    mix_fnames.append(fns)
            self.mix_fnames = mix_fnames
        else:
            self.mix_fnames = info["mix_fnames"]
        # if im_name == "im_crop_cellpose_merge":
        #     self.mix_fnames = os.listdir(self.img_dir)
        self.im_name = im_name
        
    def __len__(self):
        return len(self.mix_fnames)
    
    def impose(self, fnames):
        overlay_msk = np.zeros((self.ch_num, self.msk_size, self.msk_size), dtype=np.uint8)
        imposed = np.zeros((self.fluo_chs, self.msk_size, self.msk_size), dtype=np.int32)
        s_part = []
        b_part = []
        p_part = []
        bn_part = []
        mbn_part = []
        for fname in fnames:
            img_path = osp.join(self.img_dir, fname+".tif")
            msk_path = osp.join(self.gt_dir, fname+".tif")
            img = tif.imread(img_path)
            msk = tif.imread(msk_path)
            # img = np.array(img, dtype=np.int32)
            # # mm = img <=self.bg_thresh
            if self.bg_thresh is not None:
                mm = np.where(img <=self.bg_thresh)
                # print(mm.shape)
                img[mm] = img[mm] // self.mix_num
            imposed += img
            for i in range(self.ch_num):
                overlay_msk[i, :,:] |= msk[i, :, :]
            
            parts = fname.split("_")
            s_part.append(parts[-1])
            parts = parts[:-1]
            b_part.append(parts[1])
            p_part.append(parts[3])
            bn_part.append(parts[4])
            if len(parts) > 5:
                mbn_part.extend(parts[5:])
        
        # imposed = imposed // self.mix_num  
        name = "_".join(["B"] + b_part + ["P"] + p_part 
                        + ["b"] + bn_part + ["m"] + mbn_part+ ["s"] + s_part)
        bac_name = bn_part + mbn_part
        return imposed, overlay_msk, name, bac_name
    
    def __getitem__(self, idx):
        fnames = self.mix_fnames[idx]
        
        img, msk, fname, bn_part = self.impose(fnames)
        if self.brightness_range != (1, 1):
            brightness_factor = random.uniform(*self.brightness_range)
            img = img * brightness_factor
        if self.contrast_range != (1, 1):
            contrast_factor = random.uniform(*self.contrast_range)
            mean = np.mean(img)
            img = (img - mean) * contrast_factor + mean
        
        if self.transforms is not None:
            img, msk = self.transforms(img, msk)
            msk = msk.to(torch.long)
           
        
        bf_info ={
                "fname":fname,
                "bacn":"_".join(bn_part)
                
        }
        return img, msk, bf_info   

class BacteriaDataset(Dataset):
    def __init__(self, root_dir:str, train:bool = True, split='val', transforms=None, check_pairs= False, compress=False):
        assert osp.exists(root_dir), f"{root_dir} does not exist."
        self.root = osp.join(root_dir,)
        self.transforms = transforms
        self.compress = compress
        if compress:
            im_name = "im_16bit" 
        else:
            im_name = "im"
        
        if train:
            self.img_dir = osp.join(self.root, 'train', im_name)
            self.msk_dir = osp.join(self.root, 'train', 'gt')
        else:
            self.img_dir = osp.join(self.root, split, im_name)
            self.msk_dir = osp.join(self.root, split, 'gt')  
        assert osp.exists(self.img_dir), f"{self.img_dir} does not exist."
        assert osp.exists(self.msk_dir), f"{self.msk_dir} does not exist."
        if compress:
            img_names = [p for p in os.listdir(self.img_dir) if p.endswith('.h5')]
        else:
            img_names = [p for p in os.listdir(self.img_dir) if p.endswith('.tif')]
        msk_names = [p for p in os.listdir(self.msk_dir) if p.endswith('.png')]
        
        if check_pairs:
            new_msk_list = []
            for p in img_names:
                name = p.replace('.tif', '.png')
                if name in msk_names:
                    new_msk_list.append(name)
                else:
                    print(f"{p} has no corresponding mask file.")
            msk_names = new_msk_list

        self.img_paths = [osp.join(self.img_dir, x) for x in img_names]
        self.msk_paths = [osp.join(self.msk_dir, x) for x in msk_names]
        
    def __len__(self):
        return 10
        return len(self.img_paths)
        
    def __getitem__(self, index):
        assert index < len(self.img_paths), f"index {index} out bounds of dataset {len(self.img_paths)} "
        img_path = self.img_paths[index]
        msk_path = self.msk_paths[index % len(self.msk_paths)]
        
        if self.compress:
            with h5py.File(img_path, 'r') as f:
                img = f['fluos'][:]
        else:
            img = tif.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        img = img.transpose(1, 2, 0)
        assert img is not None, f"Failed to load image from {img_path}."
        _, h, w = img.shape
        
        msk = Image.open(msk_path)
        assert msk is not None, f"Failed to load mask from {msk_path}."
        msk = np.array(msk, dtype=np.uint8)
        
        if self.transforms is not None:
            img, msk = self.transforms(img, msk)
            msk = msk.to(torch.long)
           
        return img, msk, index
    
    @staticmethod
    def collect_fn(batch):
        imgs, msks = list(zip(*batch))
        batched_imgs = cat_list(imgs, fill_value=0)
        batched_msks = cat_list(msks, fill_value=0)
        return batched_imgs, batched_msks
    
#TODO: add test dataloader

class MixBacDataset(Dataset):
    def __init__(self, root_dir, dataset="Mix_1_Bac10", split='train', mod=False,
                 transforms=None, crop_size=512, gt_name = "gt", img_name="im_16bit",
                 crop_info_path = "/data/52HaiRuifang/Bacteria/SingleBac10/meta.json",
                 bac_class=None, collect=False):
        super().__init__()
        self.dataset_dir = osp.join(root_dir, dataset)
        self.transforms = transforms
        self.split = split
        self.crop_size = crop_size
        self.mod = mod
        self.collect = collect
        if split == "unseen":
            self.img_dir = osp.join(self.dataset_dir, 'unseen_uncrop', img_name)
            self.msk_dir = osp.join(self.dataset_dir, 'unseen_uncrop', gt_name)
        else:
            self.img_dir = osp.join(self.dataset_dir, 'seen_uncrop', img_name)
            self.msk_dir = osp.join(self.dataset_dir, 'seen_uncrop', gt_name)
        self.bf_dir = osp.join(self.dataset_dir, 'bf_72')
        
        with open(crop_info_path, 'r') as f:
            self.crop_info = json.load(f)
        crop_num = len(self.crop_info["crop_locs"])   
        split_ratio = [0.7, 0.1, 0.2]
        subsetlen = [int(crop_num * r) for r in split_ratio]
        
        if bac_class == None:
            self.ori_img_paths = glob(osp.join(self.img_dir, '*.tif'))
        else:
            self.ori_img_paths = glob(osp.join(self.img_dir, f'*_{bac_class}.tif'))
        
        if collect:
            self.all_tif, self.all_msk = self.collect_image(self.ori_img_paths, self.crop_info["crop_locs"])

        # self.ori_img_paths = self.ori_img_paths[:1]
            
        # self.subset = {
        #             "train":(0, subsetlen[0]), 
        #             "val": (subsetlen[0], subsetlen[0] + subsetlen[1]), 
        #             "test": (8, crop_num),
        #             "unseen": (8, crop_num)
        #             }    
        self.subset = {
                    "train":(0, subsetlen[0]), 
                    "val": (subsetlen[0], subsetlen[0] + subsetlen[1]), 
                    "test": (94, crop_num),
                    "unseen": (94, crop_num)
                    }  
        
    def collect_image(self, img_paths, crop_locs):
        img_list = []
        msk_list = []
        for img_path in img_paths[:]:
            # img_path = ori_img_paths[ori_index]
            iname = osp.basename(img_path)
            if iname[0] == 'w':
                fname = iname[1:]
            else:
                fname = iname
            msk_path = osp.join(self.msk_dir, fname)
            img = tif.imread(img_path)
            img = np.array(img, dtype=np.uint8)
            img = img.transpose(1, 2, 0) # h, w, ch=60
            msk = tif.imread(msk_path) # h, w, ch=10
            msk = np.array(msk, dtype=np.uint8)
            # for i in range(len(crop_locs[:4])):
            #     loc = crop_locs[i]
            #     img1 = img[loc[0]:loc[0]+self.crop_size, loc[1]:loc[1]+self.crop_size, :]
            #     msk1 = msk[loc[0]:loc[0]+self.crop_size, loc[1]:loc[1]+self.crop_size, :]
            
            #     img_list.append(img1)
            #     msk_list.append(msk1)
            img_list.append(img[:512, :512, :])
            msk_list.append(msk[:512, :512, :])
        return img_list, msk_list 
        
    def __len__(self):
        if self.collect:
            return len(self.all_tif)
        else:
            ori_num = len(self.ori_img_paths)
            crop_num = self.subset[self.split][1] - self.subset[self.split][0]
            
            return ori_num * crop_num
    
    def __getitem__(self, index):
        if self.collect:
            img = self.all_tif[index]
            msk = self.all_msk[index]
            dir = "/h3cstore_nt/ruifanghai/bacteria/u2net_full_Mix_1_Bac10_collect_wo_crop/tmp"
            tif.imwrite(osp.join(dir, f"img_{index:05d}.tif"), img.transpose(2, 0,1))
            msk = msk * 128
            tif.imwrite(osp.join(dir, f"msk_{index:05d}.tif"), msk.transpose(2, 0, 1))
            print(np.unique(msk))
        else:
            ori_index = index // (self.subset[self.split][1] - self.subset[self.split][0])
            crop_index = index % (self.subset[self.split][1] - self.subset[self.split][0])
            img_path = self.ori_img_paths[ori_index]
            iname = osp.basename(img_path)
            if iname[0] == 'w':
                fname = iname[1:]
            else:
                fname = iname
            msk_path = osp.join(self.msk_dir, fname)
            
            img = tif.imread(img_path)
            img = img.transpose(1, 2, 0) # h, w, ch=60
            msk = tif.imread(msk_path) # h, w, ch=10
            msk = np.array(msk, dtype=np.uint8)
            
            if self.mod:
                img = np.array(img, dtype=np.uint8)
            else:
                img = np.array(img, dtype=np.int32)
            
            loc = self.crop_info["crop_locs"][self.subset[self.split][0] + crop_index]
            
            img = img[loc[0]:loc[0]+self.crop_size, loc[1]:loc[1]+self.crop_size, :]
            msk = msk[loc[0]:loc[0]+self.crop_size, loc[1]:loc[1]+self.crop_size, :]
            

        if self.transforms is not None:
            img, msk = self.transforms(img, msk)
            msk = msk.to(torch.long)
        if self.split =="test" or self.split=="unseen":
            bf_path = osp.join(self.bf_dir, fname[:-4]+".png")
            bf = cv2.imread(bf_path)
            bf = bf[loc[0]:loc[0]+self.crop_size, loc[1]:loc[1]+self.crop_size, :]
            bf_info = {"iname": f"{iname[:-4]}_{crop_index:03d}",
                    "bf":bf
                    }
            return img, msk, bf_info
        else:
            return img, msk, index
    
    @staticmethod
    def collect_fn(batch):
        print(len(batch), batch[0])
        imgs, msks = list(zip(*batch))
        batched_imgs = cat_list(imgs, fill_value=0)
        batched_msks = cat_list(msks, fill_value=0)
        return batched_imgs, batched_msks
    
class MultiIterDataset(IterableDataset):
    def __init__(self, datasets, weights=None):
        super().__init__()
        self.datasets = datasets
        self.weights = weights or [1/len(datasets)] * len(datasets)
        self.total_length = sum(len(d) for d in datasets)
        
    def __iter__(self):
        iterators = [iter(d) for d in self.datasets]
        while iterators:
            chosen_dataset = random.choices(range(len(iterators)), self.weights)[0]
            
            try:
                item = next(iterators[chosen_dataset])
                yield item
            except StopIteration:
                iterators.pop(chosen_dataset)
                self.weights.pop(chosen_dataset)
                total_weight = sum(self.weights)
                self.weights = [w/total_weight for w in self.weights]
    
    def __len__(self):
        return self.total_length
    

            
        
def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images), ) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

def vis_process(processed_data, out_path, cmap='gray', title="log1p"):
    
    plt.figure(figsize=(10, 8))
    plt.imshow(processed_data, cmap=cmap)
    plt.colorbar(label='Pixel values')
    plt.title(title)
    plt.axis('on')
    plt.tight_layout()
    plt.savefig(out_path)
    
    # 打印数值范围信息
    print(f"Min value: {np.min(processed_data)}")
    print(f"Max value: {np.max(processed_data)}")
    print(f"Mean value: {np.mean(processed_data)}")
    print(f"Std deviation: {np.std(processed_data)}")

if __name__ == "__main__":
    # train_set = BacteriaDataset('/data/52HaiRuifang/Bacteria/SingleBac10')
    # val_set = BacteriaDataset('/data/52HaiRuifang/Bacteria/SingleBac10', train=False)
    
    # print(f"Train set size: {len(train_set)}; Validation set size: {len(val_set)}.")
    
    # i, m, _ = train_set.__getitem__(0)
    # print(m.shape, np.unique(m))
    # print(np.unique(i))
    # train_set = MixBacDataset('/data/52HaiRuifang/Bacteria/',
    #                           )
    # val_set = MixBacDataset('/data/52HaiRuifang/Bacteria/', 
    #                         split='val')
    root_dir = "/CX/neuro_tracking/ruifanghai/"
    ds ="global_view_20250731_new"
    compress="norm"
    # train_set = Bac11Dataset(
    #     root_dir=root_dir,
    #     dataset= ds,
    #     transforms=PresetTrain(compress=compress)
    # )
    # val_set = Bac11Dataset(
    #     root_dir=root_dir,
    #     dataset= ds,
    #     split="val",
    #     transforms=PresetEval()
    # )
    bg_thresh = 185
    train_set = Mix11Dataset(
        root_dir=root_dir,
        dataset= ds,
        mix_num=2,
        bg_thresh=bg_thresh,
        # transforms=PresetTrain(compress=compress)
    )
    val_set = Mix11Dataset(
        root_dir=root_dir,
        dataset= ds,
        split="val",
        bg_thresh=bg_thresh,
        transforms=PresetEval()
    )
    
    print(f"Train set size: {len(train_set)}; Validation set size: {len(val_set)}.")
    out_dir = f"/CX/neuro_tracking/ruifanghai/global_view_20250731_new/mix_test_bg_{bg_thresh}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    i, m, fname = train_set.__getitem__(4)
    fn = fname["fname"]
    tif.imwrite(out_dir + "/" + fn+"_im.tif", i)
    tif.imwrite(out_dir + "/" + fn+"_gt.tif", m)
    print(m.shape, np.unique(m))
    print(i.max(), i.min(), i.mean(), i.std())
    
    # labels = set()
    # for i, m, fn in train_set:
    #     label = np.unique(m)
    #     labels.update(label)
    
    # print(labels)
 
    
    
    

            








                



            
        

