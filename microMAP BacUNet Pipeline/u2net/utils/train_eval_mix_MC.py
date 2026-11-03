import torch
import torch.nn.functional as F
import utils.calculate as calculate
from utils.overlay_vis import OverlayViser
from utils.loss import mix_loss, multi_binary_loss, multilabel_loss, multilabel_softmax_CE
import math
import os
import os.path as osp
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil
# import torch.distributed as dist

viser = OverlayViser()
terminal_w = os.get_terminal_size().columns

def train_mix_MC_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, loss_type, 
                        gstep, print_freq=500, scaler=None, msk_type="label",
                        res_dir="/output",palette=(0,0,0,128,128,128),vis=False,
                        bg_weight=1.0, rank=0, binary_loss=False, cal_ch=None):
    
    model.train()
    train_loss = 0.0
    mae_metric = calculate.MeanAbsoluteError()
    jac_metric = calculate.Jaccard()
    
    if rank == 0:
        pbar = tqdm(data_loader,
                desc=f"Epoch {epoch}",
                total = len(data_loader),
                ncols=terminal_w )
    else:
        pbar = data_loader
    train_loss_list = []
    for step, (img, msk, bf_info) in enumerate(pbar):
        img, msk = img.to(device), msk.to(device)
        print(f"ima.shape = {ima.shape}")
        oname = bf_info["fname"]
        # print(oname, rank)
        if msk_type == "label":
            target = F.one_hot(msk, num_classes=model.n_classes).permute(0, 3, 1, 2).float()
        elif msk_type == "one-hot-wo":
            # msk = msk.permute(0, 3, 1, 2).float()
            b, c, h, w = msk.shape
            fg = msk.sum(dim=1, )
            bg = torch.where(fg==0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)).unsqueeze(1)
            target = torch.cat([bg, msk], dim=1)
        elif msk_type=="one-hot":
            target = msk.float()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(img)
            if cal_ch is None:
                # losses = [mix_loss(output, target) for output in outputs]
                # loss = sum(losses)
                ch_losses, loss, loss_bce, loss_dice = multi_binary_loss(outputs, target, 
                                                                         loss_type, bg_weight=bg_weight)
            else:
                # losses = [mix_loss(output[:, cal_ch, :, :], target[:, cal_ch, :, :]) for output in outputs]
                # loss = sum(losses)
                cal_outputs = [output[:, cal_ch, :, :].unsqueeze(1) for output in outputs]
                cal_msk = target[:, cal_ch, :, :].unsqueeze(1)
                ch_losses, loss, loss_bce, loss_dice = multi_binary_loss(cal_outputs, cal_msk, loss_type)
            
            # loss = multilabel_softmax_CE(outputs[0], msk, reduction='mean')
            # loss = multilabel_loss(outputs, msk, reduction='mean')
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            """ for name, parms in model.named_parameters():	
                if "side" in name:
                    print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                    ' -->grad_value:',parms.grad.shape) """
            optimizer.step()
        lr_scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        train_loss += loss.item()
        # train_loss_list.append(loss.item())
        # if loss > 80:
        #     print("large", step, "loss", loss.item())
        
        # update metrics
        final = outputs[0].detach()
        pred_prob = torch.sigmoid(final)
        pred_map = torch.ge(pred_prob, 0.5).float()
       
        mae = mae_metric.update(pred_map, target)
        # f1 = f1_metric.update(one_hot_map, target)
        jac, prec, rec, _ = jac_metric.update(pred_map, target)
        
        if rank == 0:
            pbar.set_postfix({
                'gstep': f"{gstep}",
                'loss': f"{loss:.4f}",
                'lr':f"{lr:.6f}",
                'MAE':f"{mae:.4f}",
                "IoU":f"{jac:.4f}",
                "Precision":f"{prec:.4f}",
                "Recall":f"{rec:.4f}"
            })  
        # if gstep % print_freq == 0 and rank == 0:
        #     writer.add_scalar("train/loss", loss, gstep)
        #     writer.add_scalar("train/loss_bce", loss_bce, gstep)
        #     writer.add_scalar("train/loss_dice", loss_dice, gstep)
        #     writer.add_scalar("train/lr", lr, gstep)
        #     loss_chs = {}
        #     for ch, ll in enumerate(ch_losses):
        #         loss_chs[f"ch{ch}"] = ll
                
        #     writer.add_scalars("train/ch_losses", loss_chs, gstep)
        
        gstep += 1   
    
    
    if vis and rank==0:
        msk_dir = osp.join(res_dir, 'pred_msk')
        if not osp.exists(msk_dir):
            os.makedirs(msk_dir)
        # vis_dir = osp.join(res_dir, 'vis')
        # if not osp.exists(vis_dir):
        #     os.makedirs(vis_dir)
        
        if epoch % 10 == 0: 
            pred_batch = pred_map.detach().cpu().numpy()
            msks = target.detach().cpu().numpy()
            b, c, h,w = pred_batch.shape
            vis_num = 2
            for pred, gt_m, oo in zip(pred_batch[-vis_num:], msks[-vis_num:], oname[-vis_num:]):
                img = np.zeros((h, w, 3))
                fname = f"pred_{epoch}_{oo}"
                # viser.impose_label(img, pred, osp.join(vis_dir, fname))
                # viser.gen_multi_label(img, pred, osp.join(msk_dir, fname))
                viser.cmpr_multi_label(img, pred, gt_m, osp.join(msk_dir, fname))
    
    mean_loss = train_loss / len(data_loader)  
    # idx = train_loss_list.index(max(train_loss_list)) 
    # print(f"max index {idx}, loss {train_loss_list[idx]}")
    # mae_metric.gather_from_all_processes()
    jac_metric.gather_from_all_processes()  
    return mean_loss, lr, gstep, mae_metric.compute(), jac_metric.compute()

def evaluate_mix_MC(model, data_loader, device, msk_type="one-hot-wo", 
                 cal_ch=None, res_dir=None, palette=None, 
                 vis=False,epoch=0):
    model.eval()
    mae_metric = calculate.MeanAbsoluteError()
    jac_metric = calculate.Jaccard()
    
    min_jac = 1.0
    max_mae = 0.0
    with torch.no_grad():
        for img, msk, bf_info in data_loader:
            img, msk = img.to(device), msk.to(device)
            oname = bf_info["fname"]
            if msk_type == "label":
                target = F.one_hot(msk, num_classes=model.n_classes).permute(0, 3, 1, 2).float()
            elif msk_type == "one-hot-wo":
                # msk = msk.permute(0, 3, 1, 2).float()
                b, c, h, w = msk.shape
                fg = msk.sum(dim=1, )
                bg = torch.where(fg==0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)).unsqueeze(1)
                target = torch.cat([bg, msk], dim=1)
            elif msk_type=="one-hot":
                target = msk.float()
            output = model(img)
            # 
            pred = torch.sigmoid(output) 
            pred = (pred >= 0.5).float()
            if cal_ch is None:
                mae = mae_metric.update(pred, target)
                jac, _, _, _ = jac_metric.update(pred, target)
                if jac < min_jac:
                    min_jac = jac
                if mae > max_mae:
                    max_mae = mae
            else:
                # cal_pred = normed_pred[:, cal_ch, :, :].unsqueeze(1)
                cal_pred = pred
                cal_msk = target[:, cal_ch, :, :].unsqueeze(1)
                mae_metric.update(cal_pred, cal_msk)
            
        # mae_metric.gather_from_all_processes()
            if vis and epoch % 10 == 0: 
                pred_map =pred
                msk_dir = osp.join(res_dir, 'pred_msk')
                if not osp.exists(msk_dir):
                    os.makedirs(msk_dir)
                # vis_dir = osp.join(res_dir, 'vis')
                # if not osp.exists(vis_dir):
                #     os.makedirs(vis_dir)
        
                pred_batch = pred_map.detach().cpu().numpy()
                msks = target.detach().cpu().numpy()
                b, c, h,w = pred_batch.shape
                vis_num = 4
                for pred, gt_m, oo in zip(pred_batch[-vis_num:], msks[-vis_num:], oname[-vis_num:]):
                    img = np.zeros((h, w, 3))
                    fname = f"pred_{epoch}_{oo}"
                    # viser.impose_label(img, pred, osp.join(vis_dir, fname))
                    # viser.gen_multi_label(img, gt_m, osp.join(vis_dir, fname))
                    viser.cmpr_multi_label(img, pred, gt_m, osp.join(msk_dir, fname))
                    

    return mae_metric.compute(), jac_metric.compute(), min_jac, max_mae
    