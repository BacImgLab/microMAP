'''
Author: Hai-rf 1224483724@qq.com
Date: 2024-11-11 21:33:49
LastEditTime: 2024-11-12 12:13:37
LastEditors: Hai-rf 1224483724@qq.com
Description: 
'''
import torch
import torch.nn.functional as F
import utils.calculate as calculate
from utils.loss import *
import math
from tqdm import tqdm
import os.path as osp
import os
import numpy as np
from PIL import Image
import cv2

def criterion(inputs, target):
    '''
    inputs: list [ tensor[b, n_classes, h, w] ]
    target: [b, n_classes,, h, w] , one-hot
    '''
    
    # ce = WeightedCrossEntropyLoss(num_classes=12)
    ce_weights = torch.tensor([0.3]+[1.0]*11, device=target.device)
    ce = nn.CrossEntropyLoss(weight=ce_weights, reduction='mean')
    ce_losses = [ce(input, target) for input in inputs]
    loss_ce = sum(ce_losses)
    # mcf_loss = MCFocalLoss()
    # losses = [mcf_loss(input, target) for input in inputs]
    # loss = sum(losses)
    # inputs_prob = [F.softmax(input, dim=1) for input in inputs]
    weight = torch.tensor([0.1]+[1.0]*11, device=target.device)
    dice = MultiClassDice(weight=weight)
    dice_losses = [dice(input, target) for input in inputs]
    loss_dice = sum(dice_losses)
    
    region = MCSLRegionPenaltyLoss(num_classes=12)
    region_losses = [region(input, target) for input in inputs]
    # loss_region = sum(region_losses)
    loss_region = torch.stack(region_losses).mean()
    loss = 0.6 * loss_ce + 0.3 * loss_dice + 0.1 * loss_region
    return loss, loss_ce, loss_dice, loss_region

def criterion_mullti(inputs, target):
    '''
    inputs: list [ tensor[b, n_classes, h, w] ]
    target: [b, n_classes,, h, w] , one-hot
    '''
    # inputs 7,B,C,H,W
    # target B,C,H,W
    losses = [one_channel_criterion(input, target) for input in inputs]
    # loss = sum(losses)
    # mcf_loss = MCFocalLoss()
    # losses = [mcf_loss(input, target) for input in inputs]
    # loss = sum(losses)
    return losses[0]

def one_channel_criterion(input, target):
    # inputs B,C,H,W
    # target B,C,H,W
    B,C,H,W = input.shape
    if len(input.shape) ==4:
        input = input.view(-1, C)
        target = torch.argmax(target, dim=1)
        target = target.view(-1)
        # weight = B*C*H*W-torch.count_nonzero(target)
        weight = torch.tensor([0.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], device=target.device)  
    loss = F.cross_entropy(input, target.long(), weight =weight,  ignore_index=0)
    return loss

def concat_images(im1, im2, mode='horizontal'):
    if mode == 'horizontal':
        # 水平拼接（左右）
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
    else:
        # 垂直拼接（上下）
        dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
    return dst

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, 
                    writer, vis, print_freq=50, scaler=None, msk_type="label",
                    res_dir="/output",palette=(0,0,0,128,128,128)):
    '''
    msk_type: "label", "one-hot", "one-hot-wo"
    '''
    color = {
        "TP":[0, 200, 0],
        "FP":[200, 50, 0],
        "FN":[0, 0, 200]
    }
    model.train()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    
    mae_metric = calculate.MeanAbsoluteError()
    jac_metric = calculate.Jaccard()
    
    train_loss = 0.0
    train_loss_ce = 0.0
    train_loss_dice = 0.0
    train_loss_region = 0.0
    # old2new = {6: 4, 5: 10, 9: 2, 3: 7, 1: 8, 10: 6, 7: 9, 2: 3, 4: 5, 8: 1}
    pbar = tqdm(data_loader,
                desc=f"Epoch {epoch}",
                total=len(data_loader),
                ncols = 120
                )
    
    for step, (img, msk, fname) in enumerate(pbar):
        img, msk = img.to(device), msk.to(device)
        # transfor msk to one-hot, [b, h, w, n_classes] -> [b, n_classes, h, w]
        if msk_type == "label":
            target = F.one_hot(msk, num_classes=model.n_classes).permute(0, 3, 1, 2).float()
        elif msk_type == "one-hot-wo":
            # msk = msk.permute(0, 3, 1, 2).float()
            b, c, h, w = msk.shape
            fg = msk.sum(dim=1, )
            bg = torch.where(fg==0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)).unsqueeze(1)
            target = torch.cat([bg, msk], dim=1)
        elif msk_type=="one-hot":
            target = msk
        # target_new = target.clone()
        # for old_label, new_label in old2new.items():
        #     target_new[target == old_label] = new_label  # 替换每个类别的标签

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(img)
            loss, loss_ce, loss_dice, loss_region = criterion(outputs, target)
            
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        train_loss += loss.item()
        train_loss_ce += loss_ce.item()
        train_loss_dice += loss_dice.item()
        train_loss_region += loss_region.item()
        
        # update metrics
        final = outputs[0].detach()
        # ma = torch.max(final)
        # mi = torch.min(final)
        # normed_pred = (final - mi) / (ma - mi) 
        pred_prob = F.softmax(final, dim=1)
        pred_map = torch.argmax(pred_prob, dim=1)
        one_hot_map = F.one_hot(pred_map, num_classes=model.n_classes).permute(0, 3, 1, 2).float()
        # target1 = msk.detach()
        mae = mae_metric.update(one_hot_map, target)
        jac, _, _, _ = jac_metric.update(one_hot_map, target)
        
        pbar.set_postfix({
            'loss': f"{loss:.4f}",
            'lr':f"{lr:.6f}",
            'MAE':f"{mae:.4f}",
            "IoU":f"{jac:.4f}"
        })    
        
    # mae_metric.gather_from_all_processes()
    
    mean_loss = train_loss / len(data_loader)
    mean_loss_ce = train_loss_ce / len(data_loader)
    mean_loss_dice = train_loss_dice / len(data_loader)
    mean_loss_region = train_loss_region / len(data_loader)
    loss_dict = {
        "loss": mean_loss,
        "loss_ce": mean_loss_ce,
        "loss_dice": mean_loss_dice,
        "loss_region": mean_loss_region
    }
    if vis:
        msk_dir = osp.join(res_dir, 'pred_msk')
        if not osp.exists(msk_dir):
            os.makedirs(msk_dir)
        vis_dir = osp.join(res_dir, 'cpr')
        if not osp.exists(vis_dir):
            os.makedirs(vis_dir)
        
        if epoch % 10 == 0: 
            pred_batch = pred_map.detach().cpu().numpy()
            msks = torch.argmax(target, dim=1)
            msks = msks.detach().cpu().numpy()
            for pred, gt, cfname in zip(pred_batch, msks, fname):
                vis_name = f"pred_{epoch}_{cfname}.png"
                pred_msk = Image.fromarray(pred.astype(np.uint8))
                pred_msk.putpalette(palette)
                # pred_msk.save(osp.join(msk_dir, fname))

                # fname = f"msk_{epoch}.png"
                gt_m = Image.fromarray(gt.astype(np.uint8))
                gt_m.putpalette(palette)
                # gt_m.save(osp.join(vis_dir, fname))
                pred_msk_rgb = pred_msk.convert('RGB')
                gt_m_rgb = gt_m.convert('RGB')
                concat_img = concat_images(pred_msk_rgb, gt_m_rgb, mode='horizontal')
                concat_img.save(osp.join(msk_dir, vis_name))
                
                h, w = gt.shape
                vis = np.zeros((h, w, 3), dtype=np.uint8)
                vis[np.logical_and(gt != 0, gt == pred)] = color['TP']
                vis[np.logical_and(gt==0, pred != 0)] = color['FP']
                vis[np.logical_and(gt!=0, pred != gt)] = color['FP']
                vis[np.logical_and(gt!=0, pred == 0)] = color['FN']
                cv2.imwrite(osp.join(vis_dir, vis_name), vis)
                
               
    return loss_dict, lr, mae_metric.compute(), jac_metric.compute()
    
def evaluate(model, data_loader, device, epoch, vis, palette,  
             msk_type="label", res_dir="/output"):
    color = {
        "TP":[0, 200, 0],
        "FP":[200, 50, 0],
        "FN":[0, 0, 200]
    }
    model.eval()
    
    mae_metric = calculate.MeanAbsoluteError()
    jac_metric = calculate.Jaccard()
    # old2new = {6: 4, 5: 10, 9: 2, 3: 7, 1: 8, 10: 6, 7: 9, 2: 3, 4: 5, 8: 1}
    with torch.no_grad():
        for img, msk, fname in data_loader:
            img, msk = img.to(device), msk.to(device)
            if msk_type == "label":
                target = F.one_hot(msk, num_classes=model.n_classes).permute(0, 3, 1, 2).float()
            elif msk_type == "one-hot-wo":
                # msk = msk.permute(0, 3, 1, 2).float()
                b, c, h, w = msk.shape
                bg = torch.zeros((b, 1, h, w), device=device)
                target = torch.cat([bg, msk], dim=1)
            elif msk_type=="one-hot":
                target = msk
            
            # target_new = target.clone()
            # for old_label, new_label in old2new.items():
            #     target_new[target == old_label] = new_label  # 替换每个类别的标签
            
            output = model(img)
            # ma = torch.max(output)
            # mi = torch.min(output)
            # normed_pred = (output - mi) / (ma - mi) 
            pred_prob = F.softmax(output, dim=1)
            pred_map = torch.argmax(pred_prob, dim=1)
            one_hot_map = F.one_hot(pred_map, num_classes=model.n_classes).permute(0, 3, 1, 2).float()
            mae_metric.update(one_hot_map, target)
            jac_metric.update(one_hot_map, target)
            
        # mae_metric.gather_from_all_processes()
    if vis:
        msk_dir = osp.join(res_dir, 'pred_msk')
        if not osp.exists(msk_dir):
            os.makedirs(msk_dir)
        vis_dir = osp.join(res_dir, 'cpr')
        if not osp.exists(vis_dir):
            os.makedirs(vis_dir)
        
        if epoch % 10 == 0: 
            pred_batch = pred_map.detach().cpu().numpy()
            # msks = torch.argmax(target, dim=1)
            msks = msk.detach().cpu().numpy()
            for pred, gt, cfname in zip(pred_batch, msks, fname):
                vis_name = f"pred_{epoch}_{cfname}.png"
                pred_msk = Image.fromarray(pred.astype(np.uint8))
                pred_msk.putpalette(palette)
                # pred_msk.save(osp.join(msk_dir, fname))

                # fname = f"msk_{epoch}.png"
                gt_m = Image.fromarray(gt.astype(np.uint8))
                gt_m.putpalette(palette)
                # gt_m.save(osp.join(vis_dir, fname))
                pred_msk_rgb = pred_msk.convert('RGB')
                gt_m_rgb = gt_m.convert('RGB')
                concat_img = concat_images(pred_msk_rgb, gt_m_rgb, mode='horizontal')
                concat_img.save(osp.join(msk_dir, vis_name))
                
                h, w = gt.shape
                vis = np.zeros((h, w, 3), dtype=np.uint8)
                vis[np.logical_and(gt != 0, gt == pred)] = color['TP']
                vis[np.logical_and(gt==0, pred != 0)] = color['FP']
                vis[np.logical_and(gt!=0, pred != gt)] = color['FP']
                vis[np.logical_and(gt!=0, pred == 0)] = color['FN']
                cv2.imwrite(osp.join(vis_dir, vis_name), vis)

    return mae_metric.compute(), jac_metric.compute()

def create_lr_scheduler(optimizer, 
                        num_step, 
                        epochs,
                        constant=False,
                        warmup=True,
                        warmup_epochs=50,
                        warmup_factor=1e-1,
                        end_factor=1e-2):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    def cnst(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            return 1.0
    if constant:
        # 定义一个恒定的学习率函数
        # constant_lr = lambda epoch: 1.0  # 恒定倍率
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cnst)
    else:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
                
def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
    params_group = [{"params": [], "weight_decay": 0.},  # no decay
                    {"params": [], "weight_decay": weight_decay}]  # with decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
            params_group[0]["params"].append(param)  # no decay
        else:
            params_group[1]["params"].append(param)  # with decay

    return params_group
    
    