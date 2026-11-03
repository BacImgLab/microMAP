import os
import os.path as osp
import time 
import datetime
from typing import Union, List
import json
from tqdm import tqdm
import math
from PIL import Image
import cv2
import tifffile as tif
import numpy as np
from matplotlib import pyplot as plt
import torch

class OverlayViser(object):
    def __init__(self, wobg=False):
        # np.random.seed(144)
        # _palette = ((np.random.random(3*255)) * 0.7 + 0.3) * 255
        # self.palette = _palette.astype(uint8).tolist()
        
        # 选择colormap
        colormap_name = 'Set1'
        # 设置调色板中颜色的数量
        num_colors = 97
        # 生成调色板
        palette = plt.get_cmap(colormap_name, num_colors)
        # 提取调色板的颜色
        colors = palette(range(num_colors))
        colors = colors * 255
        print(colors, colors.shape)
        colors = colors[:, :3].reshape(-1).astype(np.uint8).tolist()
        colors = [0, 0, 0] + colors
        self.palette = colors
        
        # self.cates = {
        #         # 6: "Ci",
        #         # 5: "Cc",
        #         # 9: "LGG",
        #         # 3: "Bf",
        #         # 1: "AKK",
        #         # 10: "Pp",
        #         # 7: "Eclo",
        #         # 2: "Ba",
        #         # 4: "Bp",
        #         # 8: "Fn",
        #         # 0: "bg"
        #     }
        if wobg:
            '''
            self.cates = {0: 'AKK', 1: 'Ba', 2: 'Bf', 3: 'Bp', 4: 'Cc', 5: 'Ci',
            6: 'Eclo', 7: 'LGG', 8: 'Fp', 9: 'Ecoli', 10: 'Ef'}
            '''
            self.cates = {
                0: 'Bacteria0',
                1: 'Bacteria1',
                2: 'Bacteria2',
                3: 'Bacteria3',
                4: 'Bacteria4',
                5: 'Bacteria5',
                6: 'Bacteria6',
                7: 'Bacteria7',
                8: 'Bacteria8',
                9: 'Bacteria9',
                10: 'Bacteria10',
                11: 'Bacteria11',
                12: 'Bacteria12',
                13: 'Bacteria13',
                14: 'Bacteria14',
                15: 'Bacteria15',
                16: 'Bacteria16',
                17: 'Bacteria17',
                18: 'Bacteria18',
                19: 'Bacteria19',
                20: 'Bacteria20',
                21: 'Bacteria21',
                22: 'Bacteria22',
                23: 'Bacteria23',
                24: 'Bacteria24',
                25: 'Bacteria25',
                26: 'Bacteria26',
                27: 'Bacteria27',
                28: 'Bacteria28',
                29: 'Bacteria29',
                30: 'Bacteria30',
                31: 'Bacteria31',
                32: 'Bacteria32',
                33: 'Bacteria33',
                34: 'Bacteria34',
                35: 'Bacteria35',
                36: 'Bacteria36',
                37: 'Bacteria37',
                38: 'Bacteria38',
                39: 'Bacteria39',
                40: 'Bacteria40',
                41: 'Bacteria41',
                42: 'Bacteria42',
                43: 'Bacteria43',
                44: 'Bacteria44',
                45: 'Bacteria45',
                46: 'Bacteria46',
                47: 'Bacteria47',
                48: 'Bacteria48',
                49: 'Bacteria49',
                50: 'Bacteria50',
                51: 'Bacteria51',
                52: 'Bacteria52',
                53: 'Bacteria53',
                54: 'Bacteria54',
                55: 'Bacteria55',
                56: 'Bacteria56',
                57: 'Bacteria57',
                58: 'Bacteria58',
                59: 'Bacteria59',
                60: 'Bacteria60',
                61: 'Bacteria61',
                62: 'Bacteria62',
                63: 'Bacteria63',
                64: 'Bacteria64',
                65: 'Bacteria65',
                66: 'Bacteria66',
                67: 'Bacteria67',
                68: 'Bacteria68',
                69: 'Bacteria69',
                70: 'Bacteria70',
                71: 'Bacteria71',
                72: 'Bacteria72',
                73: 'Bacteria73',
                74: 'Bacteria74',
                75: 'Bacteria75',
                76: 'Bacteria76',
                77: 'Bacteria77',
                78: 'Bacteria78',
                79: 'Bacteria79',
                80: 'Bacteria80',
                81: 'Bacteria81',
                82: 'Bacteria82',
                83: 'Bacteria83',
                84: 'Bacteria84',
                85: 'Bacteria85',
                86: 'Bacteria86',
                87: 'Bacteria87',
                88: 'Bacteria88',
                89: 'Bacteria89',
                90: 'Bacteria90',
                91: 'Bacteria91',
                92: 'Bacteria92',
                93: 'Bacteria93',
                94: 'Bacteria94',
                95: 'Bacteria95',
            }
            '''
        else:
            
            self.cates = {0:"bg",
            1: 'AKK', 2: 'Ba', 3: 'Bf', 4: 'Bp', 5: 'Cc', 6: 'Ci', 
            7: 'Eclo', 8: 'LGG', 9: 'Fp', 10: 'Ecoli', 11: 'Ef'
        }
            '''
        else:
            self.cates = {
                0: 'bg',
                1: 'Bacteria1',
                2: 'Bacteria2',
                3: 'Bacteria3',
                4: 'Bacteria4',
                5: 'Bacteria5',
                6: 'Bacteria6',
                7: 'Bacteria7',
                8: 'Bacteria8',
                9: 'Bacteria9',
                10: 'Bacteria10',
                11: 'Bacteria11',
                12: 'Bacteria12',
                13: 'Bacteria13',
                14: 'Bacteria14',
                15: 'Bacteria15',
                16: 'Bacteria16',
                17: 'Bacteria17',
                18: 'Bacteria18',
                19: 'Bacteria19',
                20: 'Bacteria20',
                21: 'Bacteria21',
                22: 'Bacteria22',
                23: 'Bacteria23',
                24: 'Bacteria24',
                25: 'Bacteria25',
                26: 'Bacteria26',
                27: 'Bacteria27',
                28: 'Bacteria28',
                29: 'Bacteria29',
                30: 'Bacteria30',
                31: 'Bacteria31',
                32: 'Bacteria32',
                33: 'Bacteria33',
                34: 'Bacteria34',
                35: 'Bacteria35',
                36: 'Bacteria36',
                37: 'Bacteria37',
                38: 'Bacteria38',
                39: 'Bacteria39',
                40: 'Bacteria40',
                41: 'Bacteria41',
                42: 'Bacteria42',
                43: 'Bacteria43',
                44: 'Bacteria44',
                45: 'Bacteria45',
                46: 'Bacteria46',
                47: 'Bacteria47',
                48: 'Bacteria48',
                49: 'Bacteria49',
                50: 'Bacteria50',
                51: 'Bacteria51',
                52: 'Bacteria52',
                53: 'Bacteria53',
                54: 'Bacteria54',
                55: 'Bacteria55',
                56: 'Bacteria56',
                57: 'Bacteria57',
                58: 'Bacteria58',
                59: 'Bacteria59',
                60: 'Bacteria60',
                61: 'Bacteria61',
                62: 'Bacteria62',
                63: 'Bacteria63',
                64: 'Bacteria64',
                65: 'Bacteria65',
                66: 'Bacteria66',
                67: 'Bacteria67',
                68: 'Bacteria68',
                69: 'Bacteria69',
                70: 'Bacteria70',
                71: 'Bacteria71',
                72: 'Bacteria72',
                73: 'Bacteria73',
                74: 'Bacteria74',
                75: 'Bacteria75',
                76: 'Bacteria76',
                77: 'Bacteria77',
                78: 'Bacteria78',
                79: 'Bacteria79',
                80: 'Bacteria80',
                81: 'Bacteria81',
                82: 'Bacteria82',
                83: 'Bacteria83',
                84: 'Bacteria84',
                85: 'Bacteria85',
                86: 'Bacteria86',
                87: 'Bacteria87',
                88: 'Bacteria88',
                89: 'Bacteria89',
                90: 'Bacteria90',
                91: 'Bacteria91',
                92: 'Bacteria92',
                93: 'Bacteria93',
                94: 'Bacteria94',
                95: 'Bacteria95',
                96: 'Bacteria96'
            }
            '''
            #0: 'AKK', 1: 'Ba', 2: 'Bf', 3: 'Bp', 4: 'Cc', 5: 'Ci',
            #6: 'Eclo', 7: 'LGG', 8: 'Fp', 9: 'Ecoli', 10: 'Ef'
            #0:"bg",
            #1: 'AKK', 2: 'Ba', 3: 'Bf', 4: 'Bp', 5: 'Cc', 6: 'Ci', 
            #7: 'Eclo', 8: 'LGG', 9: 'Fp', 10: 'Ecoli', 11: 'Ef'
            '''

    def impose_label(self, bf, msk, vis_path):
        h, w, _ = bf.shape
        # image = cv2.cvtColor(bf, cv2.COLOR_GRAY2RGB)
        # image = np.copy(bf)
        # bf = bf.float()
        c = msk.shape[0]
        for i, bimsk in enumerate(msk[:]):
            if i==0:
                continue
            cmsk = np.zeros_like(bimsk, dtype=np.uint8)
            cmsk[bimsk != 0] = i 
            cmsk = Image.fromarray(cmsk)
            cmsk.putpalette(self.palette)
            cmsk = cmsk.convert('RGB')
            cmsk = np.array(cmsk)
            # print(f"bf.shape {bf.shape} cmsk.shape {cmsk.size} bimsk {bimsk.shape} ")
            bf[bimsk!=0] = bf[bimsk!=0]*(1-(2/(c-1))) + cmsk[bimsk!=0] * (2/(c-1))
        
        cv2.imwrite(vis_path, bf)
        
    def gen_multi_label(self, bf, msk, vis_dir):
        vismsk_dir = vis_dir
        if not osp.exists(vismsk_dir):
            os.makedirs(vismsk_dir)
        for i, bimsk in enumerate(msk[:]):
            cmsk = np.zeros_like(bimsk, dtype=np.uint8)
            cmsk[bimsk != 0] = i+1
            cmsk = Image.fromarray(cmsk)
            cmsk.putpalette(self.palette)
            cmsk = cmsk.convert('RGB')
            cmsk = np.array(cmsk)
            img = np.copy(bf)
            img[bimsk!=0] = img[bimsk!=0]*0.5 + cmsk[bimsk!=0]*0.5
            cv2.imwrite(osp.join(vismsk_dir, f"ch_{i:03d}_{self.cates[i]}.png"), img)
    
    # def cmpr_ml_mc_msk(self, msks, gts, vis_dir):
    #     color = {
    #         "TP":[0, 200, 0],
    #         "FP":[200, 50, 0],
    #         "FN":[0, 50, 220]
    #     }
        
    #     vismsk_dir = vis_dir
    #     if not osp.exists(vismsk_dir):
    #         os.makedirs(vismsk_dir)
    #     c, c+1, h, w,= msks[0].shape
    #     for i, (mcmsk, gt) in enumerate(zip(msks[:], gts)):
    #         vis = np.zeros((h, w, 3), dtype=np.uint8)
           
    
    def cmpr_multi_label(self, bf, msk, gts, vis_dir):
        color = {
            "TP":[0, 200, 0],
            "FP":[200, 50, 0],
            "FN":[0, 50, 220]
        }
        
        vismsk_dir = vis_dir
        '''
        判断输出的pred是否是二值化
        '''
        if not osp.exists(vismsk_dir):
            os.makedirs(vismsk_dir)
        h, w, _ = bf.shape
        for i, (bimsk, gt) in enumerate(zip(msk[:], gts)):
            # if i == 0:
            #     continue
            #is_binary = torch.all((bimsk == 0) | (bimsk == 1))
            #print("====================================================Is the 3D array binary (only contains 0 and 1)?", is_binary)
            bimsk = np.array(bimsk != 0, dtype=int)
            gt_temp = np.array(gt != 0, dtype=int)
            #gt_temp = ()
            
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            
            vis[np.logical_and(gt_temp != 0, gt_temp == bimsk)] = color['TP']
            vis[np.logical_and(gt_temp ==0, bimsk != 0)] = color['FP']
            vis[np.logical_and(gt_temp !=0, bimsk != gt_temp)] = color['FP']
            vis[np.logical_and(gt_temp !=0, bimsk == 0)] = color['FN']
            
            # img = np.copy(bf)
            # img[bimsk!=0] = img[bimsk!=0]*0.5 + cmsk[bimsk!=0]*0.5
            cv2.imwrite(osp.join(vismsk_dir, f"ch_{i:03d}_{self.cates[i]}.png"), vis)

    
    def vis_msk(self, msk, bf, vis_path):
        cmsk = Image.fromarray(msk.astype(np.uint8))
        cmsk.putpalette(self.palette)
        cmsk = cmsk.convert('RGB')
        cmsk = np.array(cmsk)
        bf[msk!=0] = bf[msk!=0] *0.5 + cmsk[msk!=0]*0.5
        cv2.imwrite(vis_path, bf) 
      
def main(args):
    vis_dir = "/h3cstore_nt/ruifanghai/bacteria/mix_process/Mix_2_vis"
    bf_dir = "/data/52HaiRuifang/Bacteria/Mix_2_Bac10/bf_72"
    
    msk_dir = "/data/52HaiRuifang/Bacteria/Mix_2_Bac10/seen_uncrop/gt"
    
    viser = OverlayViser()
    # get fname, gen fpath
    msk_list = os.listdir(msk_dir)
    msk_list.sort()
    for fname in msk_list:
        bf_path = osp.join(bf_dir, fname.split('.')[0] + ".png")
        bf = cv2.imread(bf_path)
        msk_path = osp.join(msk_dir, fname)
        msk = tif.imread(msk_path)
        msk = np.array(msk)
        msk = msk.transpose(2, 0, 1)
        fg = msk.sum(axis=0,)
        bg = np.where(fg==0, np.array(1.0), np.array(0.0))
        bg = np.expand_dims(bg, axis=0)
        target = np.concatenate([bg, msk], axis=0)
        vispath = osp.join(vis_dir, "vis_" + fname.split('.')[0] + ".png")
        viser.impose_label(bf, target, vispath)
        
        
    
    # load img
        
if __name__ == "__main__":
    args = []
    main(args)