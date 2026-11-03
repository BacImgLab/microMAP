import torch
import torch.nn.functional as F
import utils.calculate as calculate
from utils.overlay_vis import OverlayViser
from utils.loss import mix_loss, multi_binary_loss,MultiClassHybridLoss
import math
import os
import os.path as osp
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil
# import torch.distributed as dist

viser = OverlayViser(wobg=False)
terminal_w = os.get_terminal_size().columns
#细菌种类为96时更改
mc_ml_loss = MultiClassHybridLoss(num_classes=97)
'''
bac_cates  = {
              0: 'AKK', 1: 'Ba', 2: 'Bf', 3: 'Bp', 4: 'Cc', 5: 'Ci',
                6: 'Eclo', 7: 'LGG', 8: 'Fp', 9: 'Ecoli', 10: 'Ef'
           }
'''


def check_gradients(model, step_name):
    total_grad_norm = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            if grad_norm > 0:
                print(f"{step_name} - {name}: grad_norm = {grad_norm:.6f}")
    print(f"{step_name} - Total grad norm: {total_grad_norm:.6f}")
    return total_grad_norm

bac_cates  = {
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

bacs = {val: key for key, val in bac_cates.items()}

def train_mix_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, loss_type, 
                        gstep, print_freq=500, scaler=None, msk_type="label",vis_ch_aggre=False,
                        res_dir="/output",palette=(0,0,0,128,128,128),vis=False,
                        bg_weight=1.0, rank=0, binary_loss=False, cal_ch=None):
    
    model.train()
    train_loss = 0.0
    mae_metric = calculate.MeanAbsoluteError()
    jac_metric = calculate.Jaccard()
    #print("msk = ", msk_type)
    #print("vis_ch_aggre = ", vis_ch_aggre)
    #print("cal_ch = ", cal_ch)
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
        #print(f"ima.shape = {img.shape}")
        msk = (msk > 0).float()
        #max_value = torch.max(msk)
        #print(f"max_value = {max_value}") 
        # bnames = bf_info["bacn"]
        oname = bf_info["fname"]
        bacn = bf_info["bacn"]
        optimizer.zero_grad()
        #grad_norm1 = check_gradients(model, "After first zero_grad")
        
        if msk_type == "label":
            #target = F.one_hot(msk, num_classes=model.n_classes).permute(0, 3, 1, 2).float()
            # msk = msk.permute(0, 3, 1, 2).float()
            #print(f"msk_type = {msk_type}")
            b, c, h, w = msk.shape
            fg = msk.sum(dim=1, )
            bg = torch.where(fg==0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)).unsqueeze(1)
            target = torch.cat([bg, msk], dim=1)
        elif msk_type == "one-hot-wo":
            # msk = msk.permute(0, 3, 1, 2).float()
            #print(f"msk_type = {msk_type}")
            b, c, h, w = msk.shape
            fg = msk.sum(dim=1, )
            bg = torch.where(fg==0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)).unsqueeze(1)
            target = torch.cat([bg, msk], dim=1)
            #print(f"{target[0,56,:,:]}")
        elif msk_type=="one-hot":
            target = msk.float()
            b, c, h, w = msk.shape
            class_msk = torch.zeros((b, c), device=device, dtype=torch.float32)
            for i in range(b):
                cates = [bacs[bn] for bn in bacn[i].split("_")]
                class_msk[i, cates] = 1
        elif msk_type == "ml_mc":
            b, c, h, w = msk.shape
            msk = msk.float()
            target = torch.zeros((b, c, c+1 , h, w), device=device)
            batch_indices = torch.arange(b).unsqueeze(1).expand(-1, c).flatten()
            channel_indices = torch.arange(c).unsqueeze(0).expand(b, -1).flatten()
            third_dim_indices = (torch.arange(c) + 1).unsqueeze(0).expand(b, -1).flatten()

            target[batch_indices, channel_indices, third_dim_indices] = msk.view(-1, h, w)
            # 计算背景通道 (c+1 维度的第 0 维)
            target[:, :, 0, :, :] = 1 - torch.sum(msk, dim=1, keepdim=True)
            target = target.view(b*c, -1, h, w)
            target = torch.argmax(target, dim=1)
            #print(f"target.shape = {target.shape}")
            
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            if vis_ch_aggre and epoch <1 :
                vis_dirs = [osp.join(res_dir, "ch_aggre", f"ep_{epoch}_{oo}") for oo in oname]
                outputs = model(img, vis_dirs, class_msk)
            elif vis_ch_aggre:
                outputs = model(img, class_msk = class_msk)
            else:
                outputs = model(img)
            #print(f"outputs.shape = {outputs.shape}")
            #outputs = torch.stack(outputs)
            #outputs = outputs[:, :, 0:51 :, :]
            #print(f"outputs.shape = {outputs.shape}")
            if cal_ch is None and msk_type == "one-hot-wo":
                # losses = [mix_loss(output, target) for output in outputs]
                # loss = sum(losses)
                ch_losses, loss, loss_bce, loss_dice = multi_binary_loss(outputs, target, 
                                                                         loss_type, msk_type, bg_weight=bg_weight)
                #print(f"ch_losses = {ch_losses}, loss = {loss}, loss_bce = {loss_bce}, loss_dice = {loss_dice}")
            elif cal_ch is None and msk_type == "label":
                ch_losses, loss, loss_bce, loss_dice = multi_binary_loss(outputs, target, 
                                                                         loss_type, msk_type, bg_weight=bg_weight)
            elif cal_ch is None and msk_type == "ml_mc":
                loss, loss_ce, loss_dice, loss_region = mc_ml_loss(outputs, target,) 
                                                                         
            else:
                # losses = [mix_loss(output[:, cal_ch, :, :], target[:, cal_ch, :, :]) for output in outputs]
                # loss = sum(losses)
                cal_outputs = [output[:, cal_ch, :, :].unsqueeze(1) for output in outputs]
                cal_msk = target[:, cal_ch, :, :].unsqueeze(1)
                ch_losses, loss, loss_bce, loss_dice = multi_binary_loss(cal_outputs, cal_msk, loss_type)
            
            # loss = multilabel_softmax_CE(outputs[0], msk, reduction='mean')
            # loss = multilabel_loss(outputs, msk, reduction='mean')
        
        #optimizer.zero_grad()
        #grad_norm1 = check_gradients(model, "After second zero_grad")
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            #grad_norm3 = check_gradients(model, "Before backward")
            loss.backward()
            #grad_norm4 = check_gradients(model, "After backward")
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
        if msk_type == "one-hot-wo":
            final = outputs[0].detach()
            pred_prob = torch.sigmoid(final)
            pred_map = torch.ge(pred_prob, 0.5).float()
        else:
            final = outputs[0].detach()
            prob = torch.softmax(final, dim=1)  # (b*c, c+1, h, w)-> (b*c, c+1, h, w)
            argmax_indices = torch.argmax(prob, dim=1, keepdim=True) # (b*c, 1, h, w)
            pred_map = torch.zeros_like(prob)                         # (b*c, c+1, h, w)
            pred_map.scatter_(1, argmax_indices, 1)                   # 最大值位置为1，其余为0
            #argmax_indices = torch.argmax(prob, dim=1)  # (b*c, h, w)
            #mcmsk = torch.zeros_like(prob)  # (b*c, c+1, h, w)
            #mcmsk.scatter_(1, argmax_indices.unsqueeze(1), 1)  # (b*c, c+1, h, w)
            #batch_indices = (torch.arange(b*c, device=device) % b) + 1  # (b*c,)
            #pred = mcmsk[torch.arange(b*c, device=device), batch_indices, :, :]  # (b*c, h, w)
            #pred_map = pred.view(b, c, h, w)
            #target = msk # (b, c, h, w)
       
        mae = mae_metric.update(pred_map, target)
        # f1 = f1_metric.update(one_hot_map, target)
        jac, prec, rec, _ = jac_metric.update(pred_map, target)
        
        if rank == 0:
            if msk_type == "label":
                pbar.set_postfix({
                    # 'gstep': f"{gstep}",
                    'loss': f"{loss:.4f}",
                    #'ce': f"{loss_ce:.4f}",
                    'di': f"{loss_dice:.4f}",
                    #'rg': f"{loss_region:.4f}",
                    'lr':f"{lr:.6f}",
                    'MAE':f"{mae:.4f}",
                    "IoU":f"{jac:.4f}",
                    
                })  
            else:
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'bce': f"{loss_bce:.4f}",
                    'dice': f"{loss_dice:.4f}",
                    'lr':f"{lr:.6f}",
                    'MAE':f"{mae:.4f}",
                    "IoU":f"{jac:.4f}",
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

def evaluate_mix(model, data_loader, device, rank=0, msk_type="one-hot", 
                 cal_ch=None, res_dir=None, palette=None, 
                 vis=False,epoch=0):
    model.eval()
    #model.train()
    mae_metric = calculate.MeanAbsoluteError()
    jac_metric = calculate.Jaccard()
    
    min_jac = 1.0
    max_mae = 0.0
    with torch.no_grad():
        for img, msk, bf_info in data_loader:
            img, msk = img.to(device), msk.to(device)
            msk = (msk > 0).float()
            oname = bf_info["fname"]
            if msk_type == "label":
                #target = F.one_hot(msk, num_classes=model.n_classes).permute(0, 3, 1, 2).float()
                b, c, h, w = msk.shape
                fg = msk.sum(dim=1, )
                bg = torch.where(fg==0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)).unsqueeze(1)
                target = torch.cat([bg, msk], dim=1)
            elif msk_type == "one-hot-wo":
                # msk = msk.permute(0, 3, 1, 2).float()
                b, c, h, w = msk.shape
                fg = msk.sum(dim=1, )
                bg = torch.where(fg==0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)).unsqueeze(1)
                target = torch.cat([bg, msk], dim=1)
            elif msk_type=="one-hot":
                print(f"msk_type = {msk_type}")
                target = msk.float()
            elif msk_type == "ml_mc":
                b, c, h, w = msk.shape
                msk = msk.float()
            output = model(img) # (b, c*1, h,w) or (b*c, out_ch, h, w)
            #outputs = torch.stack(outputs)
            #output = output[:, 0:51, :, :]
            #output = output[0]
            #outputs = outputs[:50]
            #print(f"output.shape = {output.shape}")
            # 
            if msk_type == "one-hot-wo":
                pred = torch.sigmoid(output) # output (b, c, h, w)-> (b, c, h, w)
                pred = (pred >= 0.5).float()
                
            else:
                final = output.detach()
                prob = torch.softmax(final, dim=1)  # (b*c, c+1, h, w)-> (b*c, c+1, h, w)
                argmax_indices = torch.argmax(prob, dim=1, keepdim=True) # (b*c, 1, h, w)
                pred = torch.zeros_like(prob)                         # (b*c, c+1, h, w)
                pred.scatter_(1, argmax_indices, 1)                   # 最大值位置为1，其余为0
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
        if vis and epoch % 50 == 0 and rank==0: 
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
    