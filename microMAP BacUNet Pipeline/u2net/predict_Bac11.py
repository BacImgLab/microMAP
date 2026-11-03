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
import pandas as pd
import csv
import torch.nn as nn

import torch
from torch.utils import data
import torch.nn.functional as F
import utils.calculate as calculate

from model import u2net_full, u2net_lite, coch_u2net_full
from utils import evaluate
from utils.viser import plot_normalized_violin, save_heatmaps, plot_all_heatmaps, save_prob
from datasets import BacteriaDataset, MixDataset, MixBacDataset, Bac11Dataset, Mix11Dataset, MixPredDataset
import transforms as T
from utils.overlay_vis import OverlayViser
from torch.utils.tensorboard import SummaryWriter
#import transforms as T

class PresetTrain:
    def __init__(self, flip_prop=2, rotate_prop=2, degree=30, compress="None",
                 scale_prop=0.5, scale_range=(0.8, 1.2),crop_size=512,
                 mean=(0, 0, 0), std=(1, 1, 1), max16=10000):
        self.transforms = T.Compose([
            T.ToTensor(max16, compress),
            # T.Resize(512),
            # T.Crop(size=(crop_size, crop_size)),
            T.Cropxy(x=68, y=68, slen = crop_size),
            # T.RandomScaleWithResize(scale_prop,scale_range),
            T.RandomHorizontalFlip(flip_prop),
            T.RandomRotate(degree, rotate_prop),
            
        ])
        
    def __call__(self, img, msk):
        return self.transforms(img, msk)


class PresetEval:
    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1),compress="0-1",
                 crop_size=512):
        self.transforms = T.Compose([
            T.ToTensor(compress=compress),
            # T.Crop((crop_size, crop_size)),
            # T.Cropxy(x=68, y=68, slen = crop_size)
            # T.Normalize(mean, std)
        ])

    def __call__(self, img, msk):
        return self.transforms(img, msk)
    
def auto_adjust_keys(model_state_dict, checkpoint_state_dict):
    # 检查模型和checkpoint的key结构是否匹配
    model_has_module = any(k.startswith('module.') for k in model_state_dict.keys())
    ckpt_has_module = any(k.startswith('module.') for k in checkpoint_state_dict.keys())
    
    # 处理四种情况：
    if model_has_module and not ckpt_has_module:
        # 当前是DDP模型但checkpoint没有前缀 → 添加前缀
        return {f'module.{k}': v for k, v in checkpoint_state_dict.items()}
    elif not model_has_module and ckpt_has_module:
        # 当前是普通模型但checkpoint有前缀 → 移除前缀
        return {k.replace('module.', '', 1): v for k, v in checkpoint_state_dict.items()}
    else:
        # 结构一致 → 直接使用
        return checkpoint_state_dict

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    total_memory = torch.cuda.get_device_properties(device).total_memory
    print(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")
    batch_size = args.batch_size
    
    res_dir = osp.join(args.output_dir, "result", "{}_{}_{}_{}".format(args.model_name, args.dataset, args.group, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    if not osp.exists(res_dir):
        os.makedirs(res_dir)
    msk_dir = osp.join(res_dir, 'pred_msk')
    if not osp.exists(msk_dir):
        os.mkdir(msk_dir)
    vis_dir = osp.join(res_dir, 'vis')
    if not osp.exists(vis_dir):
        os.mkdir(vis_dir)
    # fus_dir = osp.join(res_dir, 'fus')
    # if not osp.exists(fus_dir):
    #     os.mkdir(fus_dir)
    # input_dir = osp.join(res_dir, f'input_clip')
    # if not osp.exists(input_dir):
    #     os.mkdir(input_dir)     
    # heat_dir = osp.join(res_dir, f'heat_clip')
    # if not osp.exists(heat_dir):
    #     os.mkdir(heat_dir)
    
    # logits_dir = osp.join(res_dir, f'logits')
    # if not osp.exists(logits_dir):
    #     os.mkdir(logits_dir)
        
    prob_dir = osp.join(res_dir, f'prob')
    if not osp.exists(prob_dir):
        os.mkdir(prob_dir)
    
    probnp_dir = osp.join(res_dir, f'prob_np')
    if not osp.exists(probnp_dir):
        os.mkdir(probnp_dir)
    
    if args.msk_type == "one-hot":
        wobg = True
    else:
        wobg = False
    viser = OverlayViser(wobg=wobg)
    
    log_dir = osp.join(args.output_dir, "log", "test_{}_{}__mix{}_{}_{}".format(args.model_name, args.dataset, args.mix_num, 
                                                                                "match_h" if args.match_hist else "impose",
                                                                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    ## log writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # load
    ## model
    n_classes = args.n_classes
    if args.model_name == "u2net_full":
        model = u2net_full(img_ch = 60, n_classes=n_classes)
        #model = u2net_full(n_classes=n_classes)
    elif args.model_name == "coch_u2net_full":
        ch_cl_mapping = np.load("/data/52HaiRuifang/Bacteria/SingleBac11/ch_cl_mapping_wobg.npy")
        ch_cl_mapping[:, -1] = 0
        model = coch_u2net_full(ch_cl_mapping=ch_cl_mapping,
                                )
    else:
        model = u2net_lite(n_classes=n_classes)
        
    '''
    cates  = {
                0: 'bg',
              1: 'AKK', 2: 'Ba', 3: 'Bf', 4: 'Bp', 5: 'Cc', 6: 'Ci',
                7: 'Eclo', 8: 'LGG', 9: 'Fp', 10: 'Ecoli', 11: 'Ef'
           }
    '''
    cates = {
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
        
    
    model = model.to(device)
    ## load ckpt
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    if "model" in checkpoint:
        state_dict =  checkpoint["model"]
    else:
        state_dict =  checkpoint
    model_state_dict = model.state_dict()
    adjusted_state_dict = auto_adjust_keys(model_state_dict, state_dict)

    # 加载调整后的参数
    model.load_state_dict(adjusted_state_dict, strict=False)  # strict根据实际情况设置
    # encoder_state_dict = {k:v for k, v in state_dict.items() if "encode_modules" in k}
    # model.encode_modules.load_state_dict(encoder_state_dict, strict=False)
    # decode_state_dict = {k:v for k, v in state_dict.items() if "decode_modules" in k or "side_modules" in k or "out_conv" in k}
    # model.decode_modules.load_state_dict(decode_state_dict, strict=False)  
    
    # print("encoder",encoder_state_dict.keys() )  
    # print("decoder", decode_state_dict.keys())
    
    # # 获取模型所有参数的键
    # model_keys = set(model.state_dict().keys())
    # # 获取加载的所有参数的键
    # loaded_keys = set(encoder_state_dict.keys()).union(set(decode_state_dict.keys()))

    # # 检查未加载的参数
    # unloaded_keys = model_keys - loaded_keys
    # extra_keys = loaded_keys - model_keys

    # print("未加载的参数:", unloaded_keys)
    # print("多余的参数:", extra_keys)
      
    
 
    ## data
    ### load norm attr
    if args.data_loader== "v1": 
        test_set = BacteriaDataset(osp.join(args.data_dir, args.dataset), 
                                      train=False,
                                      transforms=PresetEval(),
                                      compress=args.compress)
        
    elif args.data_loader== "v2":
        test_set = MixBacDataset(root_dir = osp.join(args.data_dir, args.dataset),
                                   transforms=PresetEval(),)
        
    elif args.data_loader== "v3":
        test_set = MixDataset(root_dir = osp.join(args.data_dir, args.dataset),
                                   transforms=PresetEval(),)
    elif args.data_loader=="v4":
        test_set = MixPredDataset(
            root_dir=osp.join(args.data_dir, args.dataset),
            train=False,
            split=args.group,
            transforms=PresetEval(),
            tr_num=args.tr_bac_num,
            mod=False
        )
    elif args.data_loader== "v5":
        test_set = Bac11Dataset(root_dir = osp.join(args.data_dir),
                                dataset=args.dataset,
                                split=args.group,
                                #transforms=PresetTrain(compress=args.compress),
                                transforms=PresetEval(compress=args.compress),
                                tr_num=args.tr_bac_num,
                                select = args.select,
                                gt_name = args.gt_name,
                                im_name = args.im_name,
                                split_json = args.split_json
                                )
    elif args.data_loader =="v6":
        test_set = Mix11Dataset(root_dir = osp.join(args.data_dir),
                                dataset=args.dataset,
                                split = args.group,
                                transforms=PresetEval(compress=args.compress),
                                tr_num=11,
                                gt_name=args.gt_name,
                                mix_num=args.mix_num, 
                                match_hist=args.match_hist,
                                brightness = args.brightness,
                                contrast = args.contrast,
                                im_name= args.im_name,
                                save_imposed_dir='/CX/neuro_tracking/ruifanghai/global_view_20250727_new'
                                )
    elif args.data_loader== "v10":
        train_datasets = []
        val_datasets = []
        mn = args.mix_num
        dataset = Mix11Dataset(root_dir = osp.join(args.data_dir),
                                   dataset=args.dataset,
                                    split = "train",
                                    transforms=PresetTrain(compress=args.compress),
                                    tr_num=args.tr_bac_num,
                                    gt_name=args.gt_name,
                                    select=args.select,
                                    mix_num=mn,
                                    bg_thresh=args.bg_thresh,
                                    )
        train_datasets.append(dataset)
        val = Mix11Dataset(root_dir = osp.join(args.data_dir),
                               dataset=args.dataset,
                                    split="val",
                                    transforms=PresetVal(compress=args.compress),
                                    tr_num=args.tr_bac_num,
                                    gt_name=args.gt_name,
                                    select=args.select,
                                    mix_num=mn,
                                    bg_thresh=args.bg_thresh,
                                    )
        val_datasets.append(val)
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
    else:
        raise ValueError(f"data_loader {args.data_loader} not support")
        
    num_workers = min([os.cpu_count(), batch_size if batch_size>1 else 0, args.num_workers])
    test_loader = data.DataLoader(test_set,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  )
    
    # model and pred
    model.eval()
    #for module in model.modules():
    #    if isinstance(module, nn.BatchNorm2d):
    #        module.training = True  # 强制使用训练模式的行为，即使用当前批次的统计量
    mae_metric = calculate.MeanAbsoluteError()
    jac_metric = calculate.Jaccard()
    pbar = tqdm(test_loader)
    
    color = {
        "TP":[0, 200, 0],
        "FP":[200, 50, 0],
        "FN":[0, 0, 200]
    }
    
    sum_mae = 0
    sum_jac = 0
    sum_pres = 0
    sum_rec = 0
    cnt = 0
    
    metrics = {
        "mae":[],
        "jaccard":[]
    }
    
    # load origin img data
    # origin_dir = "/data/52HaiRuifang/Bacteria/10Bacteria/im"
    # with open("/data/52HaiRuifang/Bacteria/SingleBac10/meta.json", "r") as f:
    #     info = json.load(f)
    #     crop_locs = info["crop_locs"]
        
    # # Pos0_72_Ci, AKK_Pos21_00080
    # origin_bfs = {}
    # for fn in os.listdir(origin_dir):
    #     ori_im = tif.imread(osp.join(origin_dir, fn))
    #     tmp = np.array(ori_im[args.bright_field-1])
    #     ma = np.max(tmp)
    #     mi = np.min(tmp)
    #     bff = ((tmp-mi) / (ma-mi)) * 255
    #     origin_bfs[fn.split('.')[0]] = bff.astype(np.uint8)
    msk_type = args.msk_type
    idir = "/CX/neuro_tracking/ruifanghai/bacteria11_test/u2net_full_SingleBac11_mix_4_tr_mix_4/img_ch60"
    for i, (img, msk, fname) in enumerate(pbar):        
        img, msk = img.to(device), msk.to(device)
        if msk_type == "label":
            # transfor msk to one-hot, [b, h, w, n_classes] -> [b, n_classes, h, w]
            target = F.one_hot(msk, num_classes=model.n_classes).permute(0, 3, 1, 2).float()
        elif msk_type == "one-hot-wo":
            # msk = msk.permute(0, 3, 1, 2).float()
            b, c, h, w = msk.shape
            fg = msk.sum(dim=1, )
            bg = torch.where(fg==0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)).unsqueeze(1)
            target = torch.cat([bg, msk], dim=1)
        elif msk_type=="one-hot":
            target = msk.float()
            oname = fname["fname"]
            vis_dirs = [osp.join(res_dir, "ch_aggre", f"{oo}") for oo in oname]
       
        
        with torch.no_grad():
            # output = model(img, )
            # ma = torch.max(output)
            # mi = torch.min(output)
            # normed_pred = (output - mi) / (ma - mi) 
            if args.msk_type == "label":
                output = model(img, )
                pred_prob = F.softmax(output, dim=1)
                pred_map = torch.argmax(pred_prob, dim=1)
                one_hot_map = F.one_hot(pred_map, num_classes=model.n_classes).permute(0, 3, 1, 2).float()
                #这段代码对target进行二值化
                target = (target > 0).float()
                if args.ch_metric:
                    mae, ch_mae = mae_metric.update(one_hot_map, target, ch=True)
                else:
                    mae = mae_metric.update(one_hot_map, target)
                jac = jac_metric.update(one_hot_map, target)
                fnames = fname
                writer.add_scalar(f"{args.group}_MAE", mae, i)
                writer.add_scalar(f"{args.group}_IoU", jac[0], i)
                if args.ch_metric:
                    metrics["mae"].append(ch_mae)
                    metrics["jaccard"].append(jac[-1])
                sum_mae += mae
                sum_jac += jac[0]
                sum_pres += jac[1]
                sum_rec += jac[2]
                cnt += 1
            elif args.msk_type == "one-hot-wo":
                output = model(img, )
                output = output[0:51]
                #outputs = model(img, )
                #output = output[0].detach()
                pred_prob = torch.sigmoid(output)
                pred_map = torch.ge(pred_prob, 0.5).float()
                #这段代码对target进行二值化
                target = (target > 0).float()
                if args.ch_metric:
                    mae, ch_mae = mae_metric.update(pred_map, target, ch=True)
                else:
                    mae = mae_metric.update(pred_map, target)
                jac = jac_metric.update(pred_map, target)
                fnames = fname["fname"]
                writer.add_scalar(f"{args.group}_MAE", mae, i)
                writer.add_scalar(f"{args.group}_IoU", jac[0], i)
                if args.ch_metric:
                    metrics["mae"].append(ch_mae)
                    ch_jac = [-1] + jac[-1]
                    print(f"ch_jac:{ch_jac}")
                    metrics["jaccard"].append(ch_jac)
                sum_mae += mae
                sum_jac += jac[0]
                sum_pres += jac[1]
                sum_rec += jac[2]
                cnt += 1
            elif args.msk_type == "one-hot":
                output = model(img, )  
                # output = model(img, vis_dirs=vis_dirs)  
                pred_prob = torch.sigmoid(output)
                pred_map = torch.ge(pred_prob, 0.5).float()
                if args.ch_metric:
                    mae, ch_mae = mae_metric.update(pred_map, target, ch=True)
                    
                else:
                    mae = mae_metric.update(pred_map, target)
                jac = jac_metric.update(pred_map, target)
                fnames = fname["fname"]
                writer.add_scalar(f"{args.group}_MAE", mae, i)
                writer.add_scalar(f"{args.group}_IoU", jac[0], i)
                if args.ch_metric:
                    metrics["mae"].append(ch_mae)
                    ch_jac = jac[-1]
                    metrics["jaccard"].append(ch_jac)
                sum_mae += mae
                sum_jac += jac[0]
                sum_pres += jac[1]
                sum_rec += jac[2]
                cnt += 1

        # save mask
        pred_batch = pred_map.detach().cpu().numpy()
        inp = img.detach().cpu().numpy()
        logits_batch = output.detach().cpu().numpy()
        prob_batch = pred_prob.detach().cpu().numpy()
        if args.msk_type == "label":
            gt_batch = msk.detach().cpu().numpy()
        else: # args.msk_type == "one-hot-wo":
            gt_batch = target.detach().cpu().numpy()
        gt_batch = gt_batch.astype(np.uint8)
        
        h,w = pred_batch.shape[-2:]
        
        for idx, pred, gt, inimg, logits, prob in zip(fnames, pred_batch, gt_batch, inp, logits_batch, prob_batch):
            # norm_dir = osp.join(input_dir, idx)
            # os.makedirs(norm_dir, exist_ok=True)
            # plot_normalized_violin(inimg, norm_dir, sampled=True, sample_per_channel=10000)
            # plot_violin_per_channel(inimg, norm_dir,sampled=True, sample_per_channel=1000)
            ch_60 = inimg[-1,:,:]
            cv2.imwrite(osp.join(idir, idx+".png"), ch_60)
            if args.msk_type == "label":
                pred_msk = Image.fromarray(pred.astype(np.uint8))
                pred_msk.save(osp.join(msk_dir, idx+".png"))

                # vis comparison
                h, w = gt.shape
                vis = np.zeros((h, w, 3), dtype=np.uint8)

                vis[np.logical_and(gt != 0, gt == pred)] = color['TP']
                vis[np.logical_and(gt==0, pred != 0)] = color['FP']
                vis[np.logical_and(gt!=0, pred != gt)] = color['FP']
                vis[np.logical_and(gt!=0, pred == 0)] = color['FN']
                cv2.imwrite(osp.join(vis_dir, idx+".png"), vis)
                
            elif args.msk_type == "one-hot-wo":
                img = np.zeros((h, w, 3))
                viser.cmpr_multi_label(img, pred, gt, osp.join(vis_dir, idx))
                for i, pred_ch in enumerate(pred):
                    pred_msk_ch = pred_ch.astype(np.uint8)
                    ch_dir = osp.join(msk_dir, idx)
                    if not osp.exists(ch_dir):
                        os.makedirs(ch_dir)
                    cv2.imwrite(osp.join(ch_dir, f"ch_{i:03d}_{cates[i]}.png"), pred_msk_ch)
                    # pred_msk_ch.save(osp.join(msk_dir, idx, f"ch_{i:03d}_{cates[i]}.png"))
                    
                tif.imwrite(osp.join(probnp_dir, idx+".tif"), prob)
                # plot_all_heatmaps(inimg, osp.join(heat_dir, idx))
                # plot_all_heatmaps(logits, osp.join(logits_dir, idx))
                # plot_all_heatmaps(prob, osp.join(prob_dir, idx))
                # save_heatmaps(inimg, osp.join(heat_dir, idx))
                # save_heatmaps(logits, osp.join(logits_dir, idx))
                save_heatmaps(prob, osp.join(prob_dir, idx))
            elif args.msk_type == "one-hot":
                img = np.zeros((h, w, 3))
                viser.cmpr_multi_label(img, pred, gt, osp.join(msk_dir, idx))
                    
            # fus = cv2.addWeighted(bf, 0.5, vis, 0.5, 0)
            # new_img = np.zeros((h+20, w, 3), dtype=np.uint8)
            # new_img[20:, :] = fus
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(new_img, f"{fname.split('.')[0]}", (10,15), font,
            #             0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(new_img, "TP", (160, 15), font,
            #             0.5, color["TP"], 1, cv2.LINE_AA)
            # cv2.putText(new_img, "FP", (210, 15), font,
            #             0.5, color["FP"], 1, cv2.LINE_AA)
            # cv2.putText(new_img, "FN", (260, 15), font,
            #             0.5, color["FN"], 1, cv2.LINE_AA)
            
            # cv2.imwrite( osp.join(fus_dir, fname), new_img)
            
        # del output, pred_prob, pred_map, one_hot_map
        torch.cuda.empty_cache()
    if args.ch_metric:
        mae_df = pd.DataFrame(metrics["mae"])
        jac_df = pd.DataFrame(metrics["jaccard"])
        print(f"jac_df:{jac_df}")
        # ch_mean_mae = mae_df.mean(axis=0)
        # ch_mean_jac = jac_df.mean(axis=0)
        # 计算非零值的平均值
        ch_mean_mae = mae_df[mae_df != 0].mean()
        ch_mean_jac = jac_df[jac_df != 0].mean()
        
        ch_mean = pd.DataFrame([ch_mean_mae, ch_mean_jac])
        sorted_keys = sorted(args.bac_classes.keys()) # 按照bac_classes的值排序
        if args.msk_type == "one-hot-wo":
            ch_mean.columns =  ["bg"] + sorted_keys
        else:
            ch_mean.columns =  sorted_keys

        ch_mean.rows = ["MAE", "Jaccard"]
        ch_mean.to_csv(osp.join(res_dir, f"ch_mean_{args.dataset}_{args.group}.csv"), index=False)

    mean_mae = sum_mae / cnt
    mean_jac = sum_jac / cnt
    mean_pres = sum_pres / cnt
    mean_rec = sum_rec / cnt
    # 保存指标为csv 文件
    metrics = [
        ['Metric', 'Value'],
        ['Mean MAE', mean_mae],
        ['Mean Jaccard', mean_jac],
        ['Mean Precision', mean_pres],
        ['Mean Recall', mean_rec]
    ]
    with open(osp.join(res_dir, f"pred_{args.dataset}_{args.group}.csv"), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(metrics)
    print("res_dir ", res_dir)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train u2net")
    parser.add_argument("--data_dir", type=str, 
                        # default="/data/52HaiRuifang/Bacteria/",
                        default="/CX/neuro_tracking/huanggw",
                        help="root dir for a specific dataset")
    parser.add_argument("--output_dir", type=str, default="/CX/neuro_tracking/ruifanghai/bacteriaML_20251002_result")
    parser.add_argument("--dataset", type=str, default="12round11bacteria")
    parser.add_argument("--group", type=str, default="test")
    parser.add_argument("--data_loader", type=str, default="v5")
    parser.add_argument("--msk_type", type=str, default="one-hot-wo") # "label" "one-hot-wo" "one-hot"
    parser.add_argument("--n_classes", type=int, default=12)
    parser.add_argument("--mix_num", type=int, default=4)
    parser.add_argument("--split_json", type=str, default='utput.json')
    parser.add_argument("--bg_thresh", type=int, default=None, help="background thresh when mix different ims")
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt", type=str, 
                        # default = "/h3cstore_nt/ruifanghai/bacteria11_ML/mix1/crop68_u2net_full/bce_mix_1_trBac_11_lr_1.0e-04_8/ckpt/checkpoint_best.pth")
                        # default="/h3cstore_nt/ruifanghai/bacteria11_coch_ML/mix_1/ddp_crop68_coch_one-hot_norm0-1_nomsku2net_full_v6/bce_mix_1_trBac_11_lr_1.0e-04_2_nodes6_scale_1/ckpt/checkpoint_best_1670.pth")
                        # default="/h3cstore_nt/ruifanghai/bacteria11_coch_ML/mix_1/ddp_crop68_coch_one-hot_flch_norm1-1_u2net_full_v6/bce_mix_1_trBac_11_lr_1.0e-04_2_nodes6_scale_1/ckpt/checkpoint_best.pth")
                        # default = "/h3cstore_nt/ruifanghai/bacteria11_ML/mix_2/crop68_u2net_lite_None_v6/bce_mix_2_trBac_11_lr_1.0e-04_4/ckpt/checkpoint_best.pth") # ML lite
                        # default = "/h3cstore_nt/ruifanghai/bacteria11_coch_ML/mix_2/ddp_crop68_coch_one-hot_flch_norm1-1_coch_u2net_full_v6/bce_mix_2_trBac_11_lr_1.0e-04_rsm_mix1_bw0.5_bs_4__nodes4/ckpt/checkpoint_best.pth") # coch mix2
                        default = "/CX/neuro_tracking/ruifanghai/bacteriaSL_20250727_cstudy_train_new/u2net_full_v6_clip/ddp_crop68_bce_trBac_9_lr_1.0e-04_mix_num_2_ML/ckpt/checkpoint_best.pth") # SL
                        #default = "/CX/neuro_tracking/ruifanghai/bacteriaML_20250727_train_new/u2net_full_v9_clip/ddp_crop68_bce_trBac_6_lr_1.0e-04_mix_num_2_ML/ckpt/checkpoint_best.pth") # SL
                        #default = "/CX/neuro_tracking/ruifanghai/bacteriaML_20250727_train_new/u2net_full_v9_clip/ddp_crop68_bce_trBac_6_lr_1.0e-04_mix_num_4_ML/ckpt/checkpoint_best.pth") # SL clip
                        #default = "/CX/neuro_tracking/ruifanghai/bacteriaML_20250420_train_new/u2net_full_v5_clip/ddp_650_bce_trBac_0_lr_1.0e-04/ckpt/checkpoint_best.pth") # ML trmix1
                        #default = "/CX/neuro_tracking/ruifanghai/bacteriaML_20251002_train/mix_11/ddp_crop68_u2net_full_v5_clip_/CX/neuro_tracking/ruifanghai/bacteriaML_20250928_train_51bac/mix_11/ddp_crop68_u2net_full_v5_clip/bce_mix_11_trBac_9_lr_1.0e-04_rsm_mix10_bw0.05_bs_2/ckpt/checkpoint_best.pth/bce_mix_11_trBac_9_lr_1.0e-04_rsm_mix10_bw0.05_bs_2/ckpt/checkpoint_best.pth") # ML trmix4
                        #default = "/CX/neuro_tracking/ruifanghai/bacteriaML_20250925_train/mix_11/ddp_crop68_u2net_full_v5_clip_/CX/neuro_tracking/ruifanghai/bacteriaML_20250928_train_51bac/mix_11/ddp_crop68_u2net_full_v5_clip/bce_mix_11_trBac_9_lr_1.0e-04_rsm_mix10_bw0.05_bs_2/ckpt/checkpoint_best.pth/bce_mix_11_trBac_9_lr_1.0e-04_rsm_mix10_bw0.05_bs_2/ckpt/checkpoint_best.pth") #cellpose tr mix1
                        #default="/CX/neuro_tracking/ruifanghai/bacteriaML_20250922_train/mix_11/ddp_crop68_u2net_full_v5_clip_/CX/neuro_tracking/ruifanghai/bacteriaML_20250928_train_51bac/mix_11/ddp_crop68_u2net_full_v5_clip/bce_mix_11_trBac_9_lr_1.0e-04_rsm_mix10_bw0.05_bs_2/ckpt/checkpoint_best.pth/bce_mix_11_trBac_9_lr_1.0e-04_rsm_mix10_bw0.05_bs_2/ckpt/checkpoint_best.pth") #cellpose tr mix1
                        #default = "/CX/neuro_tracking/ruifanghai/bacteriaML_20250922_train/mix_11/ddp_crop68_u2net_full_v5_clip_/CX/neuro_tracking/ruifanghai/bacteriaML_20250928_train_51bac/mix_11/ddp_crop68_u2net_full_v5_clip/bce_mix_11_trBac_9_lr_1.0e-04_rsm_mix10_bw0.05_bs_2/ckpt/checkpoint_best.pth/bce_mix_11_trBac_9_lr_1.0e-04_rsm_mix10_bw0.05_bs_2/ckpt/checkpoint_best.pth") #cellpose tr mix1 none
                        #default = "/CX/neuro_tracking/ruifanghai/bacteriaML_20250907_train/u2net_full_v5_clip/ddp_crop68_bce_trBac_9_lr_1.0e-04_mix_num_11_ML/ckpt/checkpoint_best.pth") # real 60-11 cpsf
                        #default = "/CX/neuro_tracking/ruifanghai/bacteriaML_20250922_train/mix_11/ddp_crop68_u2net_full_v5_clip/bce_mix_11_trBac_9_lr_1.0e-04_rsm_mix10_bw0.05_bs_2/ckpt/checkpoint_best.pth") # real 60-11 cpsf-refine-v1
                        #default="/CX/neuro_tracking/ruifanghai/bacteriaML_20250928_train_51bac/mix_11/ddp_crop68_u2net_full_v5_clip/bce_mix_11_trBac_9_lr_1.0e-04_rsm_mix10_bw0.05_bs_2/ckpt/checkpoint_best.pth")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--tr_bac_num", type=int, default=11)
    
    parser.add_argument("--model_name", type=str, 
                        # default="u2net_lite", 
                        default="u2net_full", 
                        # default ="coch_u2net_full",
                        help="model name to indicate different network configs")
    parser.add_argument("--gt_name", type=str, default = "gt_crop")
    parser.add_argument("--im_name", type=str, default = "im_crop") # "im_crop_cellpose_merge"
    parser.add_argument("--compress", type=str, default = "clip")
    parser.add_argument("--select", type=bool, default = False)
    
    parser.add_argument("--match_hist", default=False)
    parser.add_argument("--contrast", type=float, default=0.0)
    parser.add_argument("--brightness", type=float, default=0.0)
    
    parser.add_argument("--uni", default=False)
    parser.add_argument("--ch_metric", default=True)
    
    parser.add_argument("--bright_field", type=int, default=72) # 61: 成像质量最好， 72：成像时间最接近
    
    args = parser.parse_args()
    if args.data_loader == "v5":
        if args.gt_name == "cp_sf_gt_crop":
            args.select = False
            args.output_dir = osp.join(args.output_dir, "_".join(["0907test", args.model_name, args.dataset, "crop68_u2net_full_clip_0.0001", "test_650", args.ckpt]))
        else:        
            args.output_dir = osp.join(args.output_dir, "_".join([args.model_name, args.dataset, 'SL', "crop68", "crop68_u2net_full_SingleBac11_tr11_clip_0.0001_label_5000_8"]))
    else:
        args.output_dir = osp.join(args.output_dir, "_".join([args.model_name, args.dataset, 'mix', str(args.mix_num),"tr_mix_1_cellpose", args.compress,
                                                              f"br_{args.brightness}", f"cnt_{args.contrast}" ]))
    
    args.bac_classes = {
              1: 'AKK', 2: 'Ba', 3: 'Bf', 4: 'Bp', 5: 'Cc', 6: 'Ci',
                7: 'Eclo', 8: 'LGG', 9: 'Fp', 10: 'Ecoli', 11: 'Ef'
           }
    '''
    args.bac_classes = {
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
                96: 'Bacteria95'
    }
    '''
    
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)