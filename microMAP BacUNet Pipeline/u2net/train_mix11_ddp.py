import os
import os.path as osp
import time 
import datetime
from typing import Union, List
import json
from matplotlib import pyplot as plt
import numpy as np
import logging

import torch
from torch.utils.data import DataLoader, ConcatDataset,Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import u2net_full, u2net_lite,UNet
from utils import train_mix_one_epoch, evaluate_mix, get_params_groups, create_lr_scheduler
from datasets import MixBacDataset, Mix11Dataset,Bac11Dataset, BacteriaDataset, MixDataset, MixScaleDataset, CPDataset
import transforms as T
from torch.utils.tensorboard import SummaryWriter

def get_scalar(value):
     if isinstance(value, torch.Tensor):
         return value.item()
     else:
         return value

class PresetTrain:
    def __init__(self, flip_prop=0.5, rotate_prop=0.5, degree=30, compress="None",
                 scale_prop=0.5, scale_range=(0.8, 1.2),crop_size=512,
                 mean=(0, 0, 0), std=(1, 1, 1), max16=10000):
        self.transforms = T.Compose([
            T.ToTensor(max16, compress),
            # T.Resize(512),
            # T.Crop(size=(crop_size, crop_size)),
            T.Cropxy(x=68, y=68, slen = crop_size),
            #T.RandomGaussianBlur(),
            # T.RandomScaleWithResize(scale_prop,scale_range),
            T.RandomHorizontalFlip(flip_prop),
            T.RandomRotate(degree, rotate_prop),
            
        ])
        
    def __call__(self, img, msk):
        return self.transforms(img, msk)

    
class PresetVal:
    def __init__(self, crop_size=512, compress="None",
                 mean=(0, 0, 0), std=(1, 1, 1), max16=10000):
        self.transforms = T.Compose([
            T.ToTensor(max16,compress),
            T.Cropxy(x=68, y=68, slen = crop_size),
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
    dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    total_memory = torch.cuda.get_device_properties(device).total_memory

    batch_size = args.batch_size
    
    ckpt_dir = osp.join(args.output_dir, "ckpt")
    
    log_dir = osp.join(args.output_dir, "log", "{}_mix_{}_{}_{}".format(args.model_name, 
                                                                        args.mix_num, 
                                                                        args.loss_func, 
                                                                        args.data_loader
                                                                        # datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                                                        ) )
    if dist.get_rank()==0:
        print(f"log dir is {log_dir}")
        writer = SummaryWriter(log_dir=log_dir)
        print(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")
    
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(osp.join(log_dir, "output.log")),  # 输出到文件
                logging.StreamHandler()            # 输出到屏幕
            ]
        )
    else:
        # 其他 rank 进程设为 WARNING 级别，避免冗余日志
        logging.basicConfig(level=logging.WARNING)
      
    bac_classes =  {
       'AKK':1 , 'Ba':2 , 'Bf':3 , 'Bp': 4, 'Cc':5 ,'Ci':6 , 
       'Eclo':7 , 'LGG':8 , 'Fp': 9, 'Ecoli':10 , 'Ef':11
    }
    
    if args.data_loader=="v1":
        train_dataset = BacteriaDataset(osp.join(args.data_dir, args.dataset), 
                                    transforms=PresetTrain(),
                                    compress=args.compress)
        val_dataset = BacteriaDataset(osp.join(args.data_dir, args.dataset), 
                                    transforms=PresetVal(),
                                    compress=args.compress)
    elif args.data_loader == "v2":
        train_datasets = []
        val_datasets = []
        for ds in args.datasets:
            mix_n = int(ds.split("_")[1])
            dataset = MixBacDataset(root_dir = args.data_dir,
                                    dataset = ds,
                                    transforms=PresetTrain(),
                                    gt_name="gt",
                                    bac_class = args.bac_class,
                                    )
            train_datasets.append(dataset)
            val = MixBacDataset(root_dir = args.data_dir,
                                dataset = ds,
                                split='val',
                                transforms=PresetVal(),
                                gt_name="gt",
                                bac_class = args.bac_class)
            val_datasets.append(val)
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
    elif args.data_loader == "v3":
        train_datasets = []
        val_datasets = []
        for ds in args.datasets:
            mix_n = int(ds.split("_")[1])
            dataset = MixDataset(root_dir = osp.join(args.data_dir, ds),
                                    transforms=PresetTrain(),
                                    tr_num=args.tr_bac_num,
                                    brightness=args.brightness,
                                    contrast=args.contrast)
            
            train_datasets.append(dataset)
            val = MixDataset(root_dir = osp.join(args.data_dir, ds),
                                    train=False,
                                    transforms=PresetVal(),
                                    tr_num=args.tr_bac_num)
            val_datasets.append(val)
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
    elif args.data_loader == "v4":
        train_datasets = []
        val_datasets = []
        for ds in args.datasets:
            mix_n = int(ds.split("_")[1])
            dataset = MixScaleDataset(root_dir = osp.join(args.data_dir, ds),
                                    transforms=PresetTrain(),
                                    tr_num=args.tr_bac_num,
                                    brightness=args.brightness,
                                    contrast=args.contrast,
                                    gt_name=args.gt_name,
                                    mix_num=args.mix_num)
            
            train_datasets.append(dataset)
            val = MixScaleDataset(root_dir = osp.join(args.data_dir, ds),
                                    train=False,
                                    transforms=PresetVal(),
                                    tr_num=args.tr_bac_num,
                                    gt_name=args.gt_name)
            val_datasets.append(val)
    elif args.data_loader== "v5": # for Bac11 single
        train_datasets = []
        val_datasets = []
        #print("------------------------------------------------------------现在对数据集进行初始化----------------------------------------------------")
        dataset = Bac11Dataset(root_dir = osp.join(args.data_dir),
                                split = "train",
                                transforms=PresetTrain(compress=args.in_compress),
                                #transforms=PresetVal(compress=args.in_compress),
                                tr_num=args.tr_bac_num,
                                gt_name=args.gt_name,
                                select=args.select,
                                dataset=args.dataset,
                                )
        
        train_datasets.append(dataset)
        val = Bac11Dataset(root_dir = osp.join(args.data_dir),
                                split="val",
                                transforms=PresetVal(compress=args.in_compress),
                                tr_num=args.tr_bac_num,
                                gt_name=args.gt_name,
                                select=args.select,
                                dataset=args.dataset,
                                
                                )
        val_datasets.append(val)
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
    elif args.data_loader== "v6": # for Bac11 mix1-n
        train_datasets = []
        val_datasets = []
        dataset = Bac11Dataset(root_dir = osp.join(args.data_dir),
                                split = "train",
                                transforms=PresetTrain(compress=args.in_compress),
                                tr_num=args.tr_bac_num,
                                gt_name=args.gt_name,
                                dataset=args.dataset,
                                )
        
        train_datasets.append(dataset)
        val = Bac11Dataset(root_dir = osp.join(args.data_dir),
                                split="val",
                                transforms=PresetVal(compress=args.in_compress),
                                tr_num=args.tr_bac_num,
                                gt_name=args.gt_name,
                                dataset=args.dataset,
                                )
        val_datasets.append(val)
        for mn in range(2, args.mix_num+1):
            #if mn >= 2 and mn <=7:
            #    continue
            continue
            dataset = Mix11Dataset(root_dir = osp.join(args.data_dir),
                                   dataset=args.dataset,
                                    split = "train",
                                    transforms=PresetTrain(compress=args.in_compress),
                                    tr_num=args.tr_bac_num,
                                    gt_name=args.gt_name,
                                    mix_num=mn,
                                    bg_thresh=args.bg_thresh,
                                    )
            train_datasets.append(dataset)
            val = Mix11Dataset(root_dir = osp.join(args.data_dir),
                               dataset=args.dataset,
                                    split="val",
                                    transforms=PresetVal(compress=args.in_compress),
                                    tr_num=args.tr_bac_num,
                                    gt_name=args.gt_name,
                                    mix_num=mn,
                                    bg_thresh=args.bg_thresh,
                                    )
            val_datasets.append(val)
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
    elif args.data_loader== "v7": # for Bac11 mix n
        train_datasets = []
        val_datasets = []
        for mn in range(args.mix_num, args.mix_num+1):
            dataset = Mix11Dataset(root_dir = osp.join(args.data_dir),
                                   dataset=args.dataset,
                                    split = "train",
                                    transforms=PresetTrain(compress=args.in_compress),
                                    tr_num=args.tr_bac_num,
                                    gt_name=args.gt_name,
                                    mix_num=mn,
                                    bg_thresh=args.bg_thresh,
                                    )
            train_datasets.append(dataset)
            val = Mix11Dataset(root_dir = osp.join(args.data_dir),
                               dataset=args.dataset,
                                    split="val",
                                    transforms=PresetVal(compress=args.in_compress),
                                    tr_num=args.tr_bac_num,
                                    gt_name=args.gt_name,
                                    mix_num=mn,
                                    bg_thresh=args.bg_thresh,
                                    )
            val_datasets.append(val)
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
    elif args.data_loader== "v8": # for Bac11 mix n
        train_datasets = []
        val_datasets = []
        for mn in range(args.mix_num, args.mix_num+1):
            dataset = CPDataset(root_dir = osp.join(args.data_dir),
                                dataset=args.dataset,
                                    split = "train",
                                    transforms=PresetTrain(compress=args.in_compress),
                                    tr_num=args.tr_bac_num,
                                    gt_name=args.gt_name,
                                    mix_num=mn,
                                    bg_thresh=args.bg_thresh,
                                    )
            train_datasets.append(dataset)
            val = Mix11Dataset(root_dir = osp.join(args.data_dir),
                               dataset=args.dataset,
                                    split="val",
                                    transforms=PresetVal(compress=args.in_compress),
                                    tr_num=args.tr_bac_num,
                                    gt_name=args.gt_name,
                                    mix_num=mn,
                                    bg_thresh=args.bg_thresh,
                                    )
            val_datasets.append(val)
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
    elif args.data_loader== "v9": # for Bac11 mix1-n
        train_datasets = []
        val_datasets = []
        for mn in range(args.mix_num-1, args.mix_num+1):
            dataset = Mix11Dataset(root_dir = osp.join(args.data_dir),
                                   dataset=args.dataset,
                                    split = "train",
                                    transforms=PresetTrain(compress=args.in_compress),
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
                                    transforms=PresetVal(compress=args.in_compress),
                                    tr_num=args.tr_bac_num,
                                    gt_name=args.gt_name,
                                    select=args.select,
                                    mix_num=mn,
                                    bg_thresh=args.bg_thresh,
                                    )
            val_datasets.append(val)
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
    elif args.data_loader== "v10":
        train_datasets = []
        val_datasets = []
        mn = args.mix_num
        dataset = Mix11Dataset(root_dir = osp.join(args.data_dir),
                                   dataset=args.dataset,
                                    split = "train",
                                    transforms=PresetTrain(compress=args.in_compress),
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
                                    transforms=PresetVal(compress=args.in_compress),
                                    tr_num=args.tr_bac_num,
                                    gt_name=args.gt_name,
                                    select=args.select,
                                    mix_num=mn,
                                    bg_thresh=args.bg_thresh,
                                    )
        val_datasets.append(val)
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
    
    # 分布式训练的初始化
    if dist.is_initialized():
        rank = dist.get_rank()  # 获取当前进程的 rank
        world_size = dist.get_world_size()  # 获取总的进程数

        # 使用 DistributedSampler 来确保每个进程加载不同的数据
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, args.num_workers])
    train_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   sampler=train_sampler,
                                #    collate_fn=MixBacDataset.collect_fn
                                   )
    val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True,
                                # sampler=val_sampler,
                                # collate_fn=MixBacDataset.collect_fn
                                )
    if args.bac_class is not None:
        n_classes = 1
    else:
        n_classes = args.n_classes
    if args.model_name == "u2net_full":
        model = u2net_full(img_ch = 60, n_classes=n_classes)
    elif args.model_name == "u2net_lite":
        model = u2net_lite(n_classes=n_classes)
    elif args.model_name == "unet":
        model = UNet(n_channels=60, n_classes=n_classes)
    else:
        raise NotImplementedError(f"model {args.model_name} not implemented")
    model = model.to(device)
    # for name, param in model.named_parameters():
    #     if 'weight' in name:
    #         print(f'{name}: {param.shape}')
    
    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
    if args.schedule == "cos":
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                        warmup=True, warmup_epochs=50, )
    elif args.schedule == "cnst":
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), 
                                        args.epochs, constant=True, warmup=False, warmup_epochs=10)
    else:
        raise NotImplementedError(f"schedule {args.schedule} not implemented.")
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    if args.resume:
        print(f"resume with ckpt:{args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
       
        # model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # args.start_epoch = checkpoint["epoch"] + 1
        
        if "model" in checkpoint:
            state_dict =  checkpoint["model"]
        else:
            state_dict =  checkpoint
        
        model_state_dict = model.state_dict()
        adjusted_state_dict = auto_adjust_keys(model_state_dict, state_dict)

        # 加载调整后的参数
        model.load_state_dict(adjusted_state_dict, strict=False)  # strict根据实际情况设置
        
        # if args.ddp_ckpt:
        #     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # model.load_state_dict(state_dict)
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if args.vis_train or args.vis_val:
        # 选择colormap
        colormap_name = 'tab20'
        # 设置调色板中颜色的数量
        num_colors = 20
        # 生成调色板
        palette = plt.get_cmap(colormap_name, num_colors)
        # 提取调色板的颜色
        colors = palette(range(num_colors))
        colors = colors * 255
        if dist.get_rank() == 0:
            print(colors, colors.shape)
        colors = colors[:, :3].reshape(-1).astype(np.uint8).tolist()
        colors = [0, 0, 0] + colors
        palette = colors
                
    curr_mae, curr_jac = 1.0, 0.0
    start_time = time.time()
    if args.bac_class is not None:
        cal_ch=bac_classes[args.bac_class]-1
    else:
        cal_ch= None
    sstep = 0
    if args.loss_func=='bce':
        loss_type = 'bce'
    elif args.loss_func == 'focal':
        loss_type = 'focal'             
    else:
        raise ValueError(f"loss_func {args.loss_func} not supported")
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_loader.sampler.set_epoch(epoch)
        rank = dist.get_rank()
        mean_loss, lr, estep, tr_mae, (tr_jac, tr_prec, tr_rec) = train_mix_one_epoch(model, optimizer, train_loader,
                                                       device, epoch, lr_scheduler, loss_type,
                                                       sstep, args.print_freq, scaler, 
                                                       cal_ch=cal_ch, vis=args.vis_train,
                                                       msk_type=args.msk_type, bg_weight=args.bg_weight,rank = rank , 
                                                       res_dir=osp.join(args.output_dir, "train_result"), palette= palette)
        sstep = estep
        if dist.get_rank() == 0:
            writer.add_scalar('train_loss', mean_loss, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('MAE', tr_mae, epoch)
            writer.add_scalar('IoU', tr_jac, epoch)
            writer.add_scalar('Prec', tr_prec, epoch)
            writer.add_scalar('Recall', tr_rec, epoch)
            
            save_file = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'gstep':estep,
                'args': args
            }
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
        
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            if dist.get_rank() == 0:
                if args.bac_class is not None:
                    mae, (jac, prec, rec) = evaluate_mix(model, val_loader, device, cal_ch = bac_classes[args.bac_class]-1)
                else:
                    mae, (jac, prec, rec), min_jac, max_mae = evaluate_mix(model, val_loader, device,
                                                        res_dir=osp.join(args.output_dir, "val_result"), palette=palette,
                                                        vis=args.vis_val, epoch=epoch, msk_type=args.msk_type)
                print( f"[Eval epoch:{epoch}] train_loss:{mean_loss:.4f} lr:{lr:.6f} mmae:{max_mae:.3f} mjac:{min_jac:.3f} " \
                                f"MAE:{mae:.3f} IoU:{jac.item():.3f} Prec:{prec:.3f} Rec:{rec:.3f}")

                
                writer.add_scalar('Eval_MAE', mae, epoch)
                writer.add_scalar('Eval_IoU', jac, epoch)
                writer.add_scalar('Eval_Prec', prec, epoch)
                writer.add_scalar('Eval_Recall', rec, epoch)

                if (max_mae <= curr_mae or min_jac >= curr_jac) and epoch > 0:
                    curr_mae = max_mae
                    curr_jac = min_jac
                    torch.save(save_file, os.path.join(ckpt_dir, f"checkpoint_best.pth"))
                    logging.info(f"epoch {epoch} saved best_ckpt in {ckpt_dir}.")
                    logging.info(f"max_mae: {max_mae:.3f} min_jac: {min_jac:.3f} " \
                                f"mean_mae:{mae:.3f} mean_jac:{jac:.3f}")
                if epoch > args.epochs - 5 :
                    torch.save(save_file, osp.join(ckpt_dir, f"model_{epoch}.pth"))
            if curr_mae <=0.0003 and curr_jac >= 0.98:
                break
            
    total_time = time.time() - start_time
    print( "Training time: {}".format(str(datetime.timedelta(seconds=int(total_time)))) ) 
    
    #writer.close()
    if dist.get_rank()==0:
        writer.close()
        print("TensorBoard writer closed.")
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train u2net")
    parser.add_argument("--data_dir", type=str, 
                        default="/CX/neuro_tracking/ruifanghai",
                        # default="/data/52HaiRuifang/Bacteria",
                        help="root dir for a specific dataset")
    parser.add_argument("--output_dir", type=str, default="/CX/neuro_tracking/ruifanghai/bacteriaSL_20250731_train_resultv10")
    parser.add_argument("--dataset", type=str, 
                        default="global_view_20250727_new")
                        # default="SingleBac11")
    parser.add_argument("--bac_class", type=str, default=None)
    
    parser.add_argument("--data_loader", type=str, default="v6")
    parser.add_argument("--tr_bac_num", type=int, default=4)
    parser.add_argument("--compress", type=bool, default=False)
    parser.add_argument("--msk_type", type=str, default="one-hot-wo") #"one-hot-wo", "one-hot"
    parser.add_argument("--gt_name", type=str, default="cp_sf_gt_crop")
    parser.add_argument("--im_name", type=str, default="im_crop")
    parser.add_argument("--select", type=bool, default = False)
    
    parser.add_argument("--vis_train", type=bool, default=True)
    parser.add_argument("--vis_val", type=bool, default=True)
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--contrast", type=float, default=0.0)
    parser.add_argument("--in_compress", type=str, default = "clip")
          
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--n_classes", type=int, default=97)
    parser.add_argument("--model_name", type=str, default="u2net_full", 
                        help="model name to indicate different network configs")
    parser.add_argument("--loss_func", type=str, default="bce", help="loss type")
    parser.add_argument("--bg_weight", type=float, default=1, help="background weight")
    parser.add_argument("--bg_thresh", type=int, default=None, help="background thresh when mix different ims")
    
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.0, help='weight decay (default: 1e-4)')
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-4, help='initial learning rate (default: 1e-3)')
    parser.add_argument("--schedule", type=str, default="cnst",) 
    parser.add_argument("--print_freq", default=100, type=int, help="print frequency (default: 500)")
    parser.add_argument("--eval_interval", default=10, type=int, help="print frequency (default: 10)")
    parser.add_argument("--resume", 
                            default = None,
                            #default="/h3cstore_nt/ruifanghai/bacteriaSL_20250731_train_resultv10/u2net_full_v6_clip/ddp_crop68_bce_trBac_5_lr_1.0e-04_resume_mix4/ckpt/checkpoint_best.pth",
                        #default = "/h3cstore_nt/ruifanghai/bacteriaSL_20250731_train_result/u2net_full_v5_clip/ddp_650_bce_trBac_11_lr_1.0e-04/ckpt/checkpoint_best.pth",
                        #default= "/h3cstore_nt/ruifanghai/bacteria_20250420_an_crop_reg/u2net_full_v5_clip/ddp_650_bce_trBac_11_lr_1.0e-04/ckpt/checkpoint_best.pth",                       
                        # default="/h3cstore_nt/ruifanghai/bacteria11_ML/mix_4/crop68_u2net_full_None_v6/bce_mix_4_trBac_11_lr_1.0e-04_rsm_mix3_bw0.5_bs_4/ckpt/checkpoint_best.pth", #ML_mix4
                        # default="/h3cstore_nt/ruifanghai/bacteria11_ML/mix1/crop68_u2net_full/bce_mix_1_trBac_11_lr_1.0e-04_4/ckpt/checkpoint_best.pth",
                        # default="/h3cstore_nt/ruifanghai/bacteria_ML_corre_rot/unet_bce_mix_1_trBac_1_lr_5.0e-04/ckpt/checkpoint_best.pth",
                        # default="/h3cstore_nt/ruifanghai/bacteria_ML_corre_rot/unet/bce_mix_1_trBac_9_lr_5.0e-04_resume_trbac8_/ckpt/checkpoint_best.pth",
                        
                        # default="/h3cstore_nt/ruifanghai/bacteria_ML_corre_rot/u2net_lite_bce_mix_1_trBac_1_lr_5.0e-04/ckpt/checkpoint_best.pth",
                        # default="/h3cstore_nt/ruifanghai/bacteria_ML_corre_rot/u2net_lite/bce_mix_1_trBac_10_lr_5.0e-04_resume_trbac9_/ckpt/checkpoint_best.pth",
                        
                        # default = '/h3cstore_nt/ruifanghai/bacteria_ML_corre_rot/u2net_full_bce_mix_1_trBac_1_lr_5.0e-04/ckpt/checkpoint_best.pth',
                        # default='/h3cstore_nt/ruifanghai/bacteria_ML_corre_rot/u2net_full/bce_mix_1_trBac_10_lr_5.0e-04_resume_trbac9_/ckpt/checkpoint_best.pth', 
                        type=str, help='resume from checkpoint')    
    parser.add_argument("--start_epoch", type=int, default=0, help="start epoch")
    parser.add_argument("--mix_num", type=int, default=2, help="max mix num")   
    
    parser.add_argument("--amp", action="store_true", help="use automatic mixed precision")
    parser.add_argument("--ddp_ckpt",  action="store_true")
    args = parser.parse_args()

    # args.loss_func = "MS_BCE_0.7Dice" #"MS_Focal_0.7Dice" # "BCE" #"MLCE" # _MScale_Dice"
    if args.bac_class is None:
        if args.resume:
            if args.gt_name == "cp_sf_gt_crop":
                #args.output_dir = osp.join(args.output_dir, f"{args.model_name}_{args.data_loader}_{args.in_compress}", 
                #                           f"ddp_crop68_{args.loss_func}_trBac_{args.tr_bac_num}_lr_{args.lr:.1e}_resume_mix4")
                if args.mix_num == 1: 
                    args.output_dir = osp.join(args.output_dir,f"mix_{args.mix_num}", f"ddp_crop68_{args.model_name}_{args.data_loader}_{args.in_compress}", f"{args.loss_func}_mix_{args.mix_num}_trBac_{args.tr_bac_num}_lr_{args.lr:.1e}_resume_trbac_{args.tr_bac_num-1}")
                else:
                    args.output_dir = osp.join(args.output_dir, f"{args.model_name}_{args.data_loader}_{args.in_compress}", 
                                           f"ddp_crop68_{args.loss_func}_trBac_{args.tr_bac_num}_lr_{args.lr:.1e}_mix_num_{args.mix_num}_ML")
            else:
                if args.mix_num == 1: 
                    args.output_dir = osp.join(args.output_dir,f"mix_{args.mix_num}", f"ddp_crop68_{args.model_name}_{args.data_loader}_{args.in_compress}", f"{args.loss_func}_mix_{args.mix_num}_trBac_{args.tr_bac_num}_lr_{args.lr:.1e}_resume_trbac_{args.tr_bac_num-1}")
                else:
                    args.output_dir = osp.join(args.output_dir, f"mix_{args.mix_num}", f"ddp_crop68_{args.model_name}_{args.data_loader}_{args.in_compress}", 
                                            f"{args.loss_func}_mix_{args.mix_num}_trBac_{args.tr_bac_num}_lr_{args.lr:.1e}_rsm_mix{args.mix_num-1}_bw{args.bg_weight}_bs_{args.batch_size}")
                    
        else:
            if args.gt_name == "cp_sf_gt_crop":
                args.output_dir = osp.join(args.output_dir, f"{args.model_name}_{args.data_loader}_{args.in_compress}", 
                                           f"ddp_650_{args.loss_func}_trBac_{args.tr_bac_num}_lr_{args.lr:.1e}")
            else:
                args.output_dir = osp.join(args.output_dir,f"mix_{args.mix_num}", f"ddp_crop68_{args.model_name}_{args.data_loader}_{args.in_compress}", 
                                       f"{args.loss_func}_mix_{args.mix_num}_trBac_{args.tr_bac_num}_lr_{args.lr:.1e}_{args.batch_size}")
    else:
        args.output_dir = osp.join(args.output_dir, f"{args.model_name}", f"loss_{args.loss_func}_single_{args.bac_class}_off_line_crop")
    mix_datasets = ["Mix_1_Bac10",
                     "Mix_2_Bac10",
                     "Mix_3_Bac10",
                     "Mix_4_Bac10",
                     ]
    args.datasets = mix_datasets[:args.mix_num]
    
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)