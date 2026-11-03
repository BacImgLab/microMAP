'''
Author: Hai-rf 1224483724@qq.com
Date: 2024-11-12 00:48:10
LastEditTime: 2024-11-12 00:58:02
LastEditors: Hai-rf 1224483724@qq.com
Description: 
'''
from collections import defaultdict, deque
import torch
import torch.distributed as dist
import torch.nn.functional as F
import os


def all_gather(data):
    """
    收集各个进程中的数据
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()  # 进程数
    if world_size == 1:
        return [data]

    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)

    return data_list

class MeanAbsoluteError(object):
    def __init__(self):
        self.mae_cum = 0.0
        self.cum_num = 0

    def update(self, pred: torch.Tensor, gt: torch.Tensor, ch=False):
        batch_size, c, h, w = gt.shape
        gt = (gt > 0).float()
        with torch.no_grad():
            resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
            if ch:
                ch_mae = []
                for i in range(c):
                    err = torch.sum(torch.abs(resize_pred[:, i, :, :] - gt[:, i, :, :]))
                    err = err / (batch_size * h * w)
                    ch_mae.append(err)
                total = torch.sum(torch.stack(ch_mae[1:])) + 0.1 * ch_mae[0]
                mean_error_pixels = total / c
            else:
                # mae_sum = 0.1 * torch.sum(torch.abs(resize_pred[:, 0, :, :] - gt[:, 0, :, :]), dim=(0, 1, 2)) / (batch_size * h * w)
                mae_sum = 0.0
                for i in range(1, c):
                    error_pixels = torch.sum(torch.abs(resize_pred[:, i, :, :] - gt[:, i, :, :]), dim=(0, 1, 2)) / (batch_size * h * w)
                    mae_sum+=error_pixels
                mean_error_pixels = mae_sum / (c-1)
            self.mae_cum += mean_error_pixels
            self.cum_num += 1
        if ch:
            # 转换为 Python 标量
            ch_mae_values = [err.item() for err in ch_mae]
            return mean_error_pixels, ch_mae_values
        else:
            return mean_error_pixels

    def compute(self):
        mae = self.mae_cum / self.cum_num
        return mae

    def gather_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        # 将 self.mae_cum 和 self.cum_num 转换为 torch.Tensor
        # mae_cum_tensor = torch.tensor(self.mae_cum, dtype=torch.float32, device='cuda')
        mae_cum_tensor = self.mae_cum.clone().detach().to(dtype=torch.float32, deivce="cuda")
        cum_num_tensor = torch.tensor(self.cum_num, dtype=torch.float32, device='cuda')
        
        # 同步所有进程的指标
        torch.distributed.barrier()
        torch.distributed.all_reduce(mae_cum_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(cum_num_tensor, op=torch.distributed.ReduceOp.SUM)
        
        # 将结果写回 self.mae_cum 和 self.cum_num
        self.mae_cum = mae_cum_tensor.item()
        self.cum_num = int(cum_num_tensor.item())
        

    def __str__(self):
        mae = self.compute()
        return f'MAE: {mae:.3f}'

class Jaccard(object):
    def __init__(self):
        #self.jaccard_cum = torch.zeros(1, dtype=torch.float32, device='cuda')  # 初始化为 tensor
        #self.cum_num = torch.zeros(1, dtype=torch.float32, device='cuda')      # 也建议用 tensor
        self.jaccard_cum = 0.0
        self.cum_num = 0
        self.precision_cum = 0.0
        self.recall_cum = 0.0

    def update(self, pred: torch.Tensor, gt: torch.Tensor, threshold=0.5):
        batch_size, c, h, w = gt.shape
        gt = (gt > 0).float()
        with torch.no_grad():
            resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)

            jaccard_sum = 0.0
            precision_sum = 0.0
            recall_sum = 0.0
            cnt = 0
            ch_jac = []
            for i in range(1, c):
                inter = torch.sum(torch.mul(resize_pred[:, i, :, :], gt[:, i, :, :]), dim=(0, 1, 2))
                union = torch.sum(torch.add(resize_pred[:, i, :, :], gt[:, i, :, :]), dim=(0, 1, 2))
                # pred_bin = (resize_pred > threshold)
                # gt_bin = (gt > threshold)
                # # inter = torch.sum((pred_bin & gt_bin).float())
                # # union = torch.sum((pred_bin | gt_bin).float())
                # inter = torch.sum((pred_bin[:, i, :, :] & gt_bin[:, i, :, :]).float())
                # union = torch.sum((pred_bin[:, i, :, :] | gt_bin[:, i, :, :]).float())
                gt_cnt = torch.sum(gt[:, i, :, :])
                if inter > 0 :
                    jaccard = 2 * inter / (union + 1e-4)  
                    # jaccard = inter / (union + 1e-4)  
                    precision = inter / (torch.sum(resize_pred[:, i, :, :]) + 1e-4)
                    recall = inter / (torch.sum(gt[:, i, :, :]) + 1e-4)
                    precision_sum += precision
                    recall_sum += recall
                    jaccard_sum += jaccard
                    cnt += 1
                    ch_jac.append(jaccard)
                elif inter == 0 and gt_cnt >0 :
                    ch_jac.append(0)
                    cnt += 1
                else:
                    ch_jac.append(0)
                
            if cnt > 0:
                mean_jaccard = jaccard_sum / cnt
                mean_precision = precision_sum / cnt
                mean_recall = recall_sum / cnt
                self.jaccard_cum += mean_jaccard
                self.precision_cum += mean_precision
                self.recall_cum += mean_recall
                self.cum_num += 1
            else:
                mean_jaccard = jaccard_sum
                mean_precision = precision_sum
                mean_recall = recall_sum
        ch_jac = [j.item() if j != 0 else 0.0 for j in ch_jac ]
        return mean_jaccard, mean_precision, mean_recall, ch_jac

    def compute(self):
        if self.cum_num > 0:
            jaccard = self.jaccard_cum / self.cum_num
            prec = self.precision_cum / self.cum_num
            rec = self.recall_cum / self.cum_num
        else:
            jaccard = prec = rec = 0
        return jaccard, prec, rec
    def gather_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        # 将 self.mae_cum 和 self.cum_num 转换为 torch.Tensor
        if isinstance(self.jaccard_cum, torch.Tensor):
            jac_cum_tensor = self.jaccard_cum.clone().detach().to(dtype=torch.float32, device='cuda')
        else:
            jac_cum_tensor = torch.tensor(self.jaccard_cum, dtype=torch.float32, device='cuda')
        #jac_cum_tensor = torch.tensor(self.jaccard_cum, dtype=torch.float32, device='cuda')#self.jaccard_cum.clone().detach().to(dtype=torch.float32, device='cuda')
        cum_num_tensor = torch.tensor(self.cum_num, dtype=torch.float32, device='cuda')
        
        # 同步所有进程的指标
        torch.distributed.barrier()
        torch.distributed.all_reduce(jac_cum_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(cum_num_tensor, op=torch.distributed.ReduceOp.SUM)
        
        # 将结果写回 self.mae_cum 和 self.cum_num
        self.jaccard_cum = jac_cum_tensor.item()
        self.cum_num = int(cum_num_tensor.item())

    def __str__(self):
        jaccard = self.compute()
        return f'jaccard: {jaccard:.3f}'

class F1Score(object):
    """
    refer: https://github.com/xuebinqin/DIS/blob/main/IS-Net/basics.py
    """

    def __init__(self, threshold: float = 0.5):
        self.precision_cum = None
        self.recall_cum = None
        self.num_cum = None
        self.threshold = threshold

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        batch_size, c, h, w = gt.shape
        resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        gt_num = torch.sum(torch.gt(gt, self.threshold).float())

        pp = resize_pred[torch.gt(gt, self.threshold)]  # 对应预测map中GT为前景的区域
        nn = resize_pred[torch.le(gt, self.threshold)]  # 对应预测map中GT为背景的区域

        pp_hist = torch.histc(pp, bins=255, min=0.0, max=1.0)
        nn_hist = torch.histc(nn, bins=255, min=0.0, max=1.0)

        # Sort according to the prediction probability from large to small
        pp_hist_flip = torch.flipud(pp_hist)
        nn_hist_flip = torch.flipud(nn_hist)

        pp_hist_flip_cum = torch.cumsum(pp_hist_flip, dim=0)
        nn_hist_flip_cum = torch.cumsum(nn_hist_flip, dim=0)

        precision = pp_hist_flip_cum / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
        recall = pp_hist_flip_cum / (gt_num + 1e-4)

        if self.precision_cum is None:
            self.precision_cum = torch.full_like(precision, fill_value=0.)

        if self.recall_cum is None:
            self.recall_cum = torch.full_like(recall, fill_value=0.)

        if self.num_cum is None:
            self.num_cum = torch.zeros([1], dtype=gt.dtype, device=gt.device)

        self.precision_cum += precision
        self.recall_cum += recall
        self.num_cum += batch_size

    def compute(self):
        pre_mean = self.precision_cum / self.num_cum
        rec_mean = self.recall_cum / self.num_cum
        f1_mean = (1 + 0.3) * pre_mean * rec_mean / (0.3 * pre_mean + rec_mean + 1e-8)
        max_f1 = torch.amax(f1_mean).item()
        return max_f1

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.precision_cum)
        torch.distributed.all_reduce(self.recall_cum)
        torch.distributed.all_reduce(self.num_cum)

    def __str__(self):
        max_f1 = self.compute()
        return f'maxF1: {max_f1:.3f}'
    
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)
