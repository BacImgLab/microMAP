import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def hausdorff_loss(pred_logits, target_mask):
    """
    计算 Hausdorff 距离损失 (SoftHD)
    :param pred_logits: (B, C, H, W) 预测分割概率
    :param target_mask: (B, H, W) 真实标签
    :return: 标量损失
    """
    B, C, H, W = pred_logits.shape
    loss = 0.0
    for i in range(C):
        pred_ch = pred_logits[:, i:i+1, :, :]  # (B, 1, H, W)
        pred_mask = torch.sigmoid(pred_ch)  # (B, 1, H, W)
        target_ch = target_mask[:, i:i+1, :, :]  # (B, 1, H, W)

        # 计算 Sobel 边缘
        pred_edges = sobel_filter(pred_mask) 
        target_edges = sobel_filter(target_ch)  # (B, H, W)

        # 计算距离变换 (Distance Transform)
        pred_dt = torch.cdist(pred_edges.reshape(B, -1), target_edges.reshape(B, -1))
        target_dt = torch.cdist(target_edges.reshape(B, -1), pred_edges.reshape(B, -1))

        loss += pred_dt.mean() + target_dt.mean()
    loss /= C
    return loss

def contrastive_loss(z1, z2, temperature=0.07):
    """计算 InfoNCE 对比损失"""
    z1 = F.normalize(z1, dim=-1)  # 归一化
    z2 = F.normalize(z2, dim=-1)
    batch_size = z1.shape[0]

    # 计算相似度矩阵
    logits = torch.mm(z1, z2.T) / temperature  # 余弦相似度
    labels = torch.arange(batch_size).cuda()  # 正样本索引

    loss = F.cross_entropy(logits, labels)  # InfoNCE 损失
    return loss

# **Sobel 计算函数**
def sobel_filter(tensor):
    """ 计算 PyTorch 张量的 Sobel 梯度 """
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32, device=tensor.device)
    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32, device=tensor.device)
    
    sobel_x = sobel_x.view(1, 1, 3, 3)  # 适配 Conv2D 形状
    sobel_y = sobel_y.view(1, 1, 3, 3)

    G_x = F.conv2d(tensor, sobel_x, padding=1)
    G_y = F.conv2d(tensor, sobel_y, padding=1)

    G = torch.sqrt(torch.clamp(G_x ** 2 + G_y ** 2, min=1e-6))  # 计算梯度幅值
    return G

# **损失计算**
def edge_loss(feature_map, image):
    """ 计算特征图与图像 Sobel 边缘的一致性损失 """
    # 计算特征图 Sobel 边缘
    edge_feat = sobel_filter(feature_map)

    # 计算输入图像 Sobel 边缘
    edge_image = sobel_filter(image)
    
    edge_feat = torch.clamp(edge_feat, min=1e-6, max=1.0)
    edge_image = torch.clamp(edge_image, min=1e-6, max=1.0)
    # 计算 L2 损失 (MSE Loss)
    loss = F.mse_loss(edge_feat, edge_image)
    return loss

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, background_weight=0.01):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.bg_weight = background_weight
        
    def forward(self, preds, targets):
        # 计算每个类别的像素数量
        print("Logits in loss:", preds.shape)  # 检查输入损失函数时的形状
        print("Targets in loss:", targets.shape)
        class_counts = []
        
        for i in range(self.num_classes):
            class_pixels = (targets == i).sum().float()
            class_counts.append(class_pixels)
            
        # 计算类别权重（反比于像素数量）
        total_pixels = sum(class_counts)
        class_weights = []
        
        for count in class_counts:
            if count > 0:
                # 权重与像素数量成反比
                weight = total_pixels / (self.num_classes * count)
            else:
                weight = 0.0
            class_weights.append(weight)
            # print(f"Class {i} weight: {weight}, Count: {count}, total: {total_pixels}")
            
        # 将权重转换为张量
        weights = torch.tensor(class_weights).to(preds.device)
        weights[0] *= self.bg_weight  # 背景类别的权重降低
        
        # 使用权重创建交叉熵损失
        criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean')
        loss = criterion(preds, targets)
        # print(f"wieghts:{weights}, Loss: {loss}")
        
        return loss

class MultiClassRegionPenaltyLoss(nn.Module):
    def __init__(self, num_classes, penalty_weights=None, smooth=1):
        """
        多分类场景的区域惩罚损失
        
        参数:
        - num_classes: 类别数量
        - penalty_weights: 每个类别惩罚项的权重，可以是列表或字典
        - smooth: 防止除零的平滑项
        """
        super(MultiClassRegionPenaltyLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
        
        # 设置默认的惩罚权重
        if penalty_weights is None:
            # 默认每个类别权重相等
            self.penalty_weights = {i: 5.0 for i in range(1,num_classes)}
            self.penalty_weights[0] = 0.1
        else:
            self.penalty_weights = penalty_weights
    
    def forward(self, preds, targets):
        """
        计算多分类场景的区域惩罚损失

        参数:
        - preds: 预测结果，形状为 (batch_size, num_classes, height, width)
        - targets: 目标标签，形状为 (batch_size, height, width)

        返回:
        - 损失值
        """
        b, c, h, w = preds.shape
        total_pixels = h * w
        region_penalties = [0.0,]
        probs = F.softmax(preds, dim=1)
        for i in range(1, c):
            class_penalties = []
            for bb in range(b):
                pred_i = probs[bb, i, :, :]
                pred_msk = (pred_i > 0.5).float()
                ratio = pred_msk.sum() / (total_pixels + self.smooth)
                target_i = targets[bb, :, :] == i
                target_ratio = target_i.sum() / (total_pixels + self.smooth)
                # penalty = F.relu(target_ratio - ratio)
                penalty = torch.abs(target_ratio - ratio)
                
                class_penalties.append(penalty)
                
            # 计算该类别的平均惩罚
            avg_penalty = torch.stack(class_penalties).mean() if class_penalties else torch.tensor(0.0).to(preds.device)
            
            # 应用该类别的权重
            weighted_penalty = avg_penalty * self.penalty_weights[i]
            
            region_penalties.append(weighted_penalty)
        # 汇总所有类别的惩罚
        total_region_penalty = sum(region_penalties)
        return total_region_penalty
  
class MCSLRegionPenaltyLoss(nn.Module):
    def __init__(self, num_classes, penalty_weights=None, smooth=1):
        """
        多分类场景的区域惩罚损失
        
        参数:
        - num_classes: 类别数量
        - penalty_weights: 每个类别惩罚项的权重，可以是列表或字典
        - smooth: 防止除零的平滑项
        """
        super(MCSLRegionPenaltyLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
        
        # 设置默认的惩罚权重
        if penalty_weights is None:
            # 默认每个类别权重相等
            self.penalty_weights = {i: 5.0 for i in range(1,num_classes)}
            self.penalty_weights[0] = 0.1
        else:
            self.penalty_weights = penalty_weights
    
    def forward(self, preds, targets):
        """
        计算多分类场景的区域惩罚损失

        参数:
        - preds: 预测结果，形状为 (batch_size, num_classes, height, width)
        - targets: 目标标签，形状为 (batch_size, height, width)

        返回:
        - 损失值
        """
        b, c, h, w = preds.shape
        total_pixels = h * w
        region_penalties = [0.0,]
        probs = F.softmax(preds, dim=1)  
        pred_map = torch.argmax(probs, dim=1)  
        # one_hot_map = F.one_hot(pred_map, num_classes=c).permute(0, 3, 1, 2).float()
        for i in range(1, c):
            class_penalties = []
            for bb in range(b):
                pred_msk = pred_map[bb, :, :] == i
                ratio = pred_msk.sum() / (total_pixels + self.smooth)
                target_i = targets[bb, :, :] == i
                target_ratio = target_i.sum() / (total_pixels + self.smooth)
                # penalty = F.relu(target_ratio - ratio)
                penalty = torch.abs(target_ratio - ratio)
                
                class_penalties.append(penalty)
                
            # 计算该类别的平均惩罚
            avg_penalty = torch.stack(class_penalties).mean() if class_penalties else torch.tensor(0.0).to(preds.device)
            
            # 应用该类别的权重
            weighted_penalty = avg_penalty * self.penalty_weights[i]
            
            region_penalties.append(weighted_penalty)
        # 汇总所有类别的惩罚
        total_region_penalty = sum(region_penalties)
        return total_region_penalty    
  
class MultiClassDice(nn.Module):
    def __init__(self, smooth=1.0, weight=None):
        super(MultiClassDice, self).__init__()
        self.smooth = smooth
        self.weight = weight
    
    def forward(self, preds, targets):
        b, c, h, w = preds.shape
        loss_per_class = []
        probs = F.softmax(preds, dim=1)
        for i in range(c):
            pred_i = probs[:, i, :, :]
            # target_i = targets==i
            target_i = targets[:, i, :, :]
            intersection = (pred_i * target_i).sum()
            dice_i = (2. * intersection + self.smooth) / (pred_i.sum() + target_i.sum() + self.smooth)
            loss_i = 1.0 - dice_i
            if self.weight is not None:
                loss_i = loss_i * self.weight[i]
            loss_per_class.append(loss_i)
            # print(f"Class {i}: pred_sum={pred_i.sum()}, target_sum={target_i.sum()}, "
                #   f"intersection={intersection.item()}, dice={dice_i.item()}, loss={loss_i.item()}")
        
        return torch.stack(loss_per_class).mean()
    
class MultiClassHybridLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, smooth=1.0, dice_weights=None,
                 bg_weight=0.01, penalty_weights=None):
        super(MultiClassHybridLoss, self).__init__()
        self.alpha = alpha
        self.ce_bg_weight = bg_weight
        self.smooth = smooth
        self.dice_weights = dice_weights
        self.penalty_weights = penalty_weights
        self.num_classes = num_classes
        
        self.dice_loss = MultiClassDice(smooth=smooth, 
                                        weight=dice_weights)
        self.penalty_loss = MultiClassRegionPenaltyLoss(num_classes, 
                                                        penalty_weights=penalty_weights)
        self.ce_loss = WeightedCrossEntropyLoss(num_classes, 
                                                background_weight=bg_weight)

    def forward(self, logits, targets):
        n = len(logits)
        print(f"n = {n}")
        # B, cl, h, w = logits[0].shape (b*c, c+1, h, w)

        
        total_loss_ce = 0.0
        total_loss_dice = 0.0
        total_loss_penalty = 0.0
        for j in range(n):
            logits_j = logits[j]
            loss_ce = self.ce_loss(logits_j, targets)
            loss_dice = self.dice_loss(logits_j, targets)
            loss_penalty = self.penalty_loss(logits_j, targets)
            total_loss_ce += loss_ce
            total_loss_dice += loss_dice
            total_loss_penalty += loss_penalty

        # 计算平均损失
        avg_loss_ce = total_loss_ce / n
        avg_loss_dice = total_loss_dice / n
        avg_loss_penalty = total_loss_penalty / n

        # 计算总损失
        total_loss = self.alpha * avg_loss_dice + (1 - self.alpha) * avg_loss_penalty + avg_loss_ce

        return total_loss, avg_loss_ce, avg_loss_dice, avg_loss_penalty

class MLHybridLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=0.3, beta=0.7):
        super(MLHybridLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
     
        
        
def dice_loss(pred, msk, alpha=1.0, w=1.0):
    '''
    w = 1.0, is original dice loss.
    w != 1.0, is Generalized dice loss(GDL).w is calculated by class pixels ratio.
    GDL is better than dice loss when class pixels ratio is unbalanced.
    GDL focus more on small object class.
    '''
    # flatten
    pred = pred.view(pred.shape[0], -1) # b, h*w
    msk = msk.view(msk.shape[0], -1) # b, h*w
    
    intersection = (pred * msk).sum(dim=1)
    union = pred.sum(dim=1) + msk.sum(dim=1)
    
    dice_coeff = (2 * w * intersection + alpha) / (w * union + alpha)
    
    dice_loss = 1 - dice_coeff.mean()
    return dice_loss

def weighted_BCE(pred, target, empty_weight=1.0):
    # 计算前景和背景比例
    #torch.set_printoptions(threshold=float('inf'), linewidth=160, profile="full")
    #print(f"target = {target}")
    unique_values = torch.unique(target)

    #print(f"Unique elements in the tensor: {unique_values}")
    #print(f"target含有的数值")
    foreground_ratio = target.view(target.shape[0], -1).float().mean(dim=1)
    background_ratio = 1 - foreground_ratio
    
    foreground_weight = 1 / (foreground_ratio + 1e-4)
    background_weight = 1 / (background_ratio + 1e-4)
    
    zero_fg_mask = (foreground_ratio == 0)
    foreground_weight[zero_fg_mask] = 0
    background_weight[zero_fg_mask] = empty_weight
    
    # foreground_weight = foreground_weight.view(target.shape[0], target.shape[1])
    # background_weight = background_weight.view(target.shape[0], target.shape[1])
    
    
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    has_negative_bce = torch.any(bce_loss < 0).item()
    weighted_loss = bce_loss * (
            target * foreground_weight[:, None, None, None] + 
            (1 - target) * background_weight[:, None, None, None]
        )
    has_negative = torch.any(weighted_loss < 0).item()
    #print(f"Unique elements in the tensor: {unique_values}, has_negative_bce = {has_negative_bce}")
    #print(f"has_negative = {has_negative}")
    #print(f"========================================has_negative_bce = {has_negative_bce}, target.shape = {target.shape}, bce_loss = {bce_loss}, fore = {foreground_weight}, back = {background_weight}, weight_loss = {weighted_loss}, has_negative = {has_negative}, res = {torch.mean(weighted_loss)}=============================================")
    return torch.mean(weighted_loss)

def mix_loss_new(pred, msk, alpha=1.0, gamma=0.3, beta=0.7):
    b, c, h, w = pred.shape
    loss_bce = 0.0
    loss_dice = 0.0
    ch_losses = torch.zeros((c))
    
    for i in range(c):
        #print(f"现在计算第{i}个通道上的细菌的loss")
        pred_ch = pred[:, i, :, :]
        msk_ch = msk[:, i, :, :]
        #max_val = msk_ch.max()
        #print(f"max_val = {max_val}")
        #is_binary = torch.all((pred_ch >= 0) & (pred_ch <= 1))
        #print("====================================================Is the 3D array binary (only contains 0 and 1)?", is_binary)
        # bce = F.binary_cross_entropy_with_logits(pred_ch, msk_ch,reduction='mean')
        #print(pred_ch)
        msk_ch = (msk_ch > 0).float()
        #print(f"msk = {msk}")
        #max_value = torch.max(msk_ch)
        #if max_value == 1.0:
        #    print(f"现在计算第{i}个通道上的细菌的loss")
        #print(f"max_value = {max_value}") 
        #pred_ch = (pred_ch > 0).float()
        bce = weighted_BCE(pred_ch, msk_ch)
        #bce = criterion(pred, target)
        
        pred_sigmoid = torch.sigmoid(pred_ch)
        dice = dice_loss(pred_sigmoid, msk_ch, alpha)
        
        ch_losses[i] = (gamma*bce + beta * dice) 
        loss_bce = loss_bce + bce
        loss_dice = loss_dice + dice

    # mean BCE loss & mean Dice loss
    loss_bce /= c
    loss_dice /= c 
    
    # loss = loss_bce + loss_dice
    # return loss
    return ch_losses, loss_bce, loss_dice


#def mix_loss(pred, msk, alpha=0.5, beta=0.5, smooth=1e-5):
def mix_loss(pred, msk, alpha=0.5, beta=0.5, smooth=1e-5):
    b, c, h, w = pred.shape

    # 1. 将标签转换为 one-hot 格式
    msk = torch.stack(
        [(msk[:, i] > 0) for i in range(msk.shape[1])],
        dim=1
    ).float()
    #max_value = torch.max(msk)
    #print(f"max_value = {max_value}")
    if msk.dim() == 4:
        target_onehot = msk.to(dtype=pred.dtype)
        target_idx = torch.argmax(target_onehot, dim=1)
    elif msk.dim() == 3:
        target_idx = msk.long()
        target_onehot = F.one_hot(
            target_idx, num_classes=c
        ).permute(0, 3, 1, 2).to(dtype=pred.dtype)
    else:
        raise ValueError(
            f"msk 形状应为 (B,C,H,W) 或 (B,H,W)，当前为 {msk.shape}"
        )

    # 2. 按图片公式计算交叉熵
    # L_CE = -1/(HW) * sum_k sum_ij GT_k * log(M_k)
    # softmax + log_softmax 数值稳定地实现 log(M_k)
    log_probs = F.log_softmax(pred, dim=1)
    loss_ce = -(
        target_onehot * log_probs
    ).sum(dim=1).mean()

    # 3. 获取预测概率图 M_ML
    probs = F.softmax(pred, dim=1)

    # 4. 按类别计算 Dice 系数
    # Dice_k = 2|M_k ∩ GT_k| / (|M_k| + |GT_k| + epsilon)
    intersection = (probs * target_onehot).sum(dim=(2, 3))
    denominator = (
        probs.sum(dim=(2, 3))
        + target_onehot.sum(dim=(2, 3))
    )

    dice_coeff = (
        2.0 * intersection
    ) / (denominator + smooth)

    # 对类别和 batch 求平均
    mean_dice = dice_coeff.mean()

    # 图片公式给出的是 Dice 系数，因此转为 Dice Loss
    loss_dice = 1.0 - mean_dice

    # 5. 每个类别的损失，用于保持原函数的监控输出
    ch_losses = []

    for i in range(c):
        # 该类别的交叉熵贡献
        ce_ch = -(
            target_onehot[:, i] * log_probs[:, i]
        ).mean()

        # 该类别的 Dice Loss
        inter_ch = intersection[:, i]
        denom_ch = denominator[:, i]

        dice_ch = 1.0 - (
            2.0 * inter_ch
        ) / (denom_ch + smooth)

        dice_ch = dice_ch.mean()

        ch_losses.append(alpha * ce_ch + beta * dice_ch)

    ch_losses = torch.stack(ch_losses)

    # 6. 按混合损失公式组合
    total_loss = alpha * loss_ce + beta * loss_dice

    # 保留原函数的三个返回值
    return ch_losses, loss_ce, loss_dice
    
    

def mix_focal_loss(pred, msk, alpha=1.0, gamma=0.3, beta=0.7):
    b, c, h, w = pred.shape
    loss_bce = 0.0
    loss_dice = 0.0
    ch_losses = torch.zeros((c))
    fg_focal_loss = BiFocalLoss(alpha=0.9)
    bg_focal_loss = BiFocalLoss(alpha=0.1)
    
    for i in range(c):
        pred_ch = pred[:, i, :, :]
        msk_ch = msk[:, i, :, :]
        msk_ch = (msk_ch > 0).float()
        
        # bce = F.binary_cross_entropy_with_logits(pred_ch, msk_ch,reduction='mean')
        if i==0:
            bce = bg_focal_loss(pred_ch, msk_ch)
        else:
            bce = fg_focal_loss(pred_ch, msk_ch)
        
        pred_sigmoid = torch.sigmoid(pred_ch)
        dice = dice_loss(pred_sigmoid, msk_ch, alpha)
        
        ch_losses[i] = (gamma*bce + beta * dice) 
        loss_bce += bce
        loss_dice += dice

    # mean BCE loss & mean Dice loss
    loss_bce /= c
    loss_dice /= c 
    
    # loss = loss_bce + loss_dice
    # return loss
    return ch_losses, loss_bce, loss_dice

def multi_binary_loss(outputs, msk, loss_type='bce', msk_type='one-hot-wo',bg_weight=0.1, 
                      alpha=1.0, gamma=0.3, beta=0.7, binary_loss=False, cal_ch=None):
    n = len(outputs)
    b, c,h,w = outputs[0].shape 
    #print(f"msk.shape = {msk.shape}")
    #print(f"{n}, {b}, {c}, {h}, {w}")
    all_losses = torch.zeros((n, c))
    all_bces = torch.zeros(n)
    all_dices = torch.zeros(n)
    #print(f"be_weight = {bg_weight}")
    for i, output in enumerate(outputs):
        #output = output[0:50]
        #print(f"output.shape = {output.shape}")
        if loss_type == 'focal':
            all_losses[i], all_bces[i], all_dices[i] = mix_focal_loss(output, msk)
        elif loss_type == 'bce':
            if msk_type == 'one-hot-wo':
                print(f"here is one-hot-wo")
                #print(f"output.shape = {output.shape}, msk.shape = {msk.shape}")
                all_losses[i], all_bces[i], all_dices[i] = mix_loss_new(output, msk)
                #print(f"all_losses[{i}] = {all_losses[i]}")
                #print(f"all_losses = {all_losses}, all_bces = {all_bces}, all_dices = {all_dices}")
            elif msk_type == 'label':
                all_losses[i], all_bces[i], all_dices[i] = mix_loss(output, msk)
    weights = torch.ones(c)  # 初始权重全为1
    # TODO:改变各个容易预测的种类loss所占的权重
    '''
    weights[1] = 0.2
    weights[2] = 0.5
    weights[3] = 0.2
    weights[6] = 0.5
    weights[9] = 0.2
    weights[10] = 0.2
    '''
    weights[0] = bg_weight  # 指定第一个通道的权重为bg_weight
        
    ch_losses = torch.sum(all_losses, dim=0) # loss of each ch over batch
    losses = torch.sum(all_losses * weights.view(1, -1), dim=1) / c # loss of each item in batch over chs
    loss =  torch.sum(losses)
    loss_bce = torch.sum(all_bces)
    loss_dice = torch.sum(all_dices)
    return ch_losses, loss, loss_bce, loss_dice

def multilabel_softmax_CE(pred, label, mask_zero=False, reduction='none'):
    """
    Multilabel Categorical CrossEntropy for 4D inputs.

    Args:
        pred: Logits tensor with shape [b, c, h, w], without activation.
        label: One-hot tensor with shape [b, c, h, w], containing 0 or 1.
        mask_zero: If True, masks zero-padding in labels.
        reduction: Specifies the reduction to apply ('none', 'mean', 'sum').

    Returns:
        Loss tensor.
    """
    if mask_zero:
        # Ensure logits have a padding channel for masking
        zeros = torch.zeros_like(pred[:, :1, :, :])  # Shape [b, 1, h, w]
        label = torch.cat([label, zeros], dim=1)  # Extend channel for padding class

        infs = torch.ones_like(zeros) * torch.inf
        pred_ce = torch.cat([infs, pred_ce[:, 1:, :, :]], dim=1)
    
    pred = (1 - 2 * label) * pred
    # Calculate losses for positive and negative labels
    pos_pred = pred - (1 - label) * 1e12  # Positive part: only active for true labels
    neg_pred = pred - label * 1e12   # Negative part: only active for false labels
    zeros = torch.zeros_like(pred[:, :1, :, :])  # Shape [b, 1, h, w]
    neg_pred = torch.cat([neg_pred, zeros], dim=1)  # Extend channel for padding class
    pos_pred = torch.cat([pos_pred, zeros], dim=1)  # Extend channel for padding class

    # Positive loss: log-sum-exp over positive predictions
    pos_loss = torch.logsumexp(pos_pred, dim=1)  # Sum over channel dimension

    # Negative loss: log-sum-exp over negative predictions
    neg_loss = torch.logsumexp(neg_pred, dim=1)  # Sum over channel dimension

    # Total loss
    loss = pos_loss + neg_loss  # Shape [b, h, w]

    # Reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unexpected reduction {reduction}")

def multilabel_loss(outputs, msk, reduction='mean', alpha=1.0):
    n = len(outputs)
    b, c,h,w = outputs[0].shape 
    all_losses = torch.zeros((n,))

    for j, output in enumerate(outputs):
        all_losses[j] = multilabel_softmax_CE(output, msk, reduction=reduction)
        loss_dice = 0
        for i in range(c):
            pred_ch = output[:, i, :, :]
            msk_ch = msk[:, i, :, :]
            pred_sigmoid = torch.sigmoid(pred_ch)
            dice = dice_loss(pred_sigmoid, msk_ch, alpha)
            loss_dice += dice
        
        loss_dice /= c
        all_losses[j] += loss_dice
    
    loss = torch.sum(all_losses)
    return loss


class MCFocalLoss(nn.Module):
    '''
    Multiple classes Focal loss
    The output for each pixel contains only one class selected from all the classes.
    '''
    def __init__(self, alpha=0.25, gamma=1, reduction='mean'):
        super(MCFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        '''
        inputs: [B, C, H, W]
        targets: [B, H, W]
        '''
        probs = F.softmax(inputs, dim=1)
        targets = targets.view(-1, 1)
        pt = probs.view(-1, probs.size(1)).gather(1, targets)
        
        focal_weight =  (1 - pt) ** self.gamma 
        
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            alpha_t = self.alpha.gather(0, targets.view(-1))
            
        loss = - alpha_t * focal_weight * torch.log(pt)
        
        if self.reduction == 'mean': 
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unexpected reduction {self.reduction}")
        
class BiFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(BiFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算输入的预测概率
        inputs = torch.sigmoid(inputs)  # 对输入进行sigmoid变换，得到预测概率
        targets = targets.float()  # 将目标转换为float类型
        
        # 计算交叉熵损失
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # 计算p_t
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        
        # 计算Focal Loss
        loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss
        
        # 对损失进行处理
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss