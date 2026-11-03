from typing import Union, List
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from matplotlib import pyplot as plt
import os.path as osp
import os
import numpy as np

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.2)
        
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            init.zeros_(self.conv.bias)
        
        
    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # 打印输入形状
        x = self.conv(x)
        # print(f"After conv shape: {x.shape}")  # 打印卷积后张量形状
        x = self.bn(x)
        # print(f"After bn shape: {x.shape}")  # 打印批标准化后张量形状
        x = self.relu(x)
        # print(f"After relu shape: {x.shape}")  # 打印 ReLU 后张量形状
        x = self.dropout(x)
        # return self.relu(self.bn(self.conv(x)))
        return x

class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, flag = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True) # ceil_mode=True, to keep the same size as the original paper
        return super().forward(x)
    
class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, flag = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.up_flag: # upsample the feature map
            x = F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False)
        return super().forward(torch.cat([x, y], dim=1))
    
class RSU(nn.Module):
    def __init__(self, height, in_ch, mid_ch, out_ch):
        super().__init__()
        
        assert height >=2, "height should be greater than or equal to 2"

        # single conv
        self.in_conv = ConvBNReLU(in_ch, out_ch)
        
        # repeat layers
        encode_list = [ConvBNReLU(out_ch, mid_ch)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))
            
        # bridge layer
        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.in_conv(x)
        
        x = x_in
        x_list = []
        for module in self.encode_modules[:-1]:
            x = module(x)
            x_list.append(x)

        x = self.encode_modules[-1](x)
        
        for module, x2 in zip(self.decode_modules, reversed(x_list)):
            x = module(x, x2)

        return x + x_in
    
class RSU4F(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        
        self.encode_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([
            ConvBNReLU(out_ch, mid_ch),
            ConvBNReLU(mid_ch, mid_ch, dilation=2),
            ConvBNReLU(mid_ch, mid_ch, dilation=4),
            ConvBNReLU(mid_ch, mid_ch, dilation=8),
        ])
        self.decode_modules = nn.ModuleList([
           ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
           ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
           ConvBNReLU(mid_ch * 2, out_ch),
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.encode_in(x)

        x = x_in
        x_list = []
        for module in self.encode_modules[:-1]:
            x = module(x)
            x_list.append(x)
            
        x = self.encode_modules[-1](x)
        
        for module, x2 in zip(self.decode_modules, reversed(x_list)):
            x = module(torch.cat([x, x2], dim=1))

        return x + x_in
    
class Coch_U2NET(nn.Module):
    def __init__(self, cfg: dict, n_classes: int, 
                 ch_cl_mapping, scale_aggregate=True,
                 in_ch: int = 1,out_ch: int = 1
                 ):
        '''
        ch_cl_mapping: [cl_num, ch_num], e.g. [12, 60]
        '''
        assert "encode" in cfg, "cfg should contain 'encode' key"
        assert "decode" in cfg, "cfg should contain 'decode' key"
        self.encode_num = len(cfg["encode"])
        
        super().__init__()
        
        self.n_classes = n_classes
        self.cates = {
                0: 'AKK', 1: 'Ba', 2: 'Bf', 3: 'Bp', 4: 'Cc', 5: 'Ci',
                6: 'Eclo', 7: 'LGG', 8: 'Fp', 9: 'Ecoli', 10: 'Ef'
            }
        if scale_aggregate:
            ch_cl_mapping = self.scale_ch_cl_mapping(ch_cl_mapping)
            print("scaled ch_cl_mapping:", ch_cl_mapping)
        ch_cl_mapping = torch.from_numpy(ch_cl_mapping).float()
        
        self.register_buffer('ch_cl_mapping', ch_cl_mapping) 
        assert n_classes == ch_cl_mapping.shape[0]
        self.out_ch = out_ch
        self.scale_aggregate = scale_aggregate

        encode_list = []
        decode_list = []
        side_list = []
        
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            encode_list.append( RSU4F(*c[1:4]) if c[4] else RSU(*c[:4]) )
            if c[5]:
                side_list.append(nn.Conv2d(c[3], self.out_ch, kernel_size=3, padding=1))

        for c in cfg["decode"]:
            decode_list.append( RSU4F(*c[1:4]) if c[4] else RSU(*c[:4]) )
            if c[5]:
                side_list.append(nn.Conv2d(c[3], self.out_ch, kernel_size=3, padding=1))
                
        self.encode_modules = nn.ModuleList(encode_list) # [enc1, enc2, enc3, enc4, enc5, enc6 as bridge]
        self.decode_modules = nn.ModuleList(decode_list) # [dec5, dec4, dec3, dec2, dec1]
        self.side_modules = nn.ModuleList(side_list)
        
        # self.out_conv1 = nn.Conv2d(self.encode_num * n_classes, n_classes*2, kernel_size=1)
        # self.out_conv2 = nn.Conv2d(n_classes*2, n_classes, kernel_size=1)
        # self.relu = nn.ReLU(replace=True)
        
        # init.kaiming_normal_(self.out_conv1.weight, mode='fan_out', nonlinearity='relu')
        # if self.out_conv1.bias is not None:
        #     init.zeros_(self.out_conv1.bias)
        # init.kaiming_normal_(self.out_conv2.weight, mode='fan_out', nonlinearity='relu')
        # if self.out_conv2.bias is not None:
        #     init.zeros_(self.out_conv2.bias)
        self.out_conv = nn.Conv2d(self.encode_num * self.out_ch, self.out_ch, kernel_size=1)
        init.kaiming_normal_(self.out_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.out_conv.bias is not None:
            init.zeros_(self.out_conv.bias)
            
    def scale_ch_cl_mapping(self, mapping, scale=1):
        ones_count_per_row = np.sum(mapping, axis=1)
        ones_count_per_row = np.where(ones_count_per_row == 0, 1, ones_count_per_row) # 避免除以0
        scale_factor = scale / ones_count_per_row
        scaled_mapping = mapping * scale_factor[:, np.newaxis]
        return scaled_mapping
            
    def visualize_features(self, features, vdir):
        
        num_features = features.shape[0]
        h, w = features.shape[1:]
        max_values = []
        min_values = []
        rows = int(num_features ** 0.5) # 计算子图的行数和列数
        cols = (num_features + rows - 1) // rows

        fig_chs, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()
        all_max = features.max().item()
        all_min = features.min().item()
        for i in range(num_features):
            feature = features[i].squeeze().cpu().detach().numpy()
            im = axes[i].imshow(feature, cmap='hot', vmin=all_min, vmax=all_max)
            axes[i].axis('off')
            axes[i].set_title(f"{i}_{self.cates[i]}")
            # 添加colorbar
            cbar = fig_chs.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)  # 调整colorbar刻度大小
            max_values.append(feature.max().item())
            min_values.append(feature.min().item())
        for j in range(num_features, rows * cols): # 隐藏多余的子图
            axes[j].axis('off')
        plt.tight_layout()
        
        fig_max_min = plt.figure(figsize=(10, 6))
        plt.plot(range(num_features), max_values, label='Max Values', marker='o')
        plt.plot(range(num_features), min_values, label='Min Values', marker='x')
        plt.xlabel('FLuos Channels')
        plt.ylabel('Value')
        plt.title('Max and Min Values of Normed and Ch-Sum aggregation')
        
        # 设置横坐标刻度标签为类别名称
        category_names = [f"{i}_{self.cates[i]}" for i in range(num_features)]
        plt.xticks(range(num_features), category_names, rotation=45)
        
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        os.makedirs(vdir, exist_ok=True)
        fig_chs.savefig(osp.join(vdir, 'ch_sum.png'), bbox_inches='tight', dpi=150)
        fig_max_min.savefig(osp.join(vdir, 'max_min.png'), bbox_inches='tight', dpi=150)
        
        plt.close(fig_chs)
        plt.close(fig_max_min)
        
    
        
    def forward(self, x: torch.Tensor, vis_dirs=None, class_msk=None) -> Union[torch.Tensor, List[torch.Tensor]]:
        '''
        x的预期形状: [batch_size, channels, height, width],其中channels=self.ch_num, e.g.60
        '''
        b, c, h, w, = x.shape
        # 通道强度归一化
        x_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0].min(dim=-3, keepdim=True)[0] # 计算最小值
        x_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=-3, keepdim=True)[0] # 和最大值
        x = (x - x_min) / (x_max - x_min + 1e-6) # 线性归一化到 [0, 1]
        # x = 2 * x - 1 # 线性归一化到 [-1, 1]
        
        
        # 为每一类创建加权通道
        weighted_channels = []
        for category in range(self.n_classes):
            # 提取当前类别的通道权重 [60]
            category_weights = self.ch_cl_mapping[category]         
            category_weights = category_weights.view(1, -1, 1, 1)  # 扩展权重维度以匹配输入张量 [1, 60, 1, 1]
            weighted_input = x * category_weights # 对输入进行加权 [batch_size, 60, height, width] × [1, 60, 1, 1]
            weighted_sum = weighted_input.sum(dim=1, keepdim=True) # 沿通道维度求和，得到单个通道 [batch_size, 1, height, width]
            weighted_channels.append(weighted_sum)
        
        # 将所有加权通道拼接在一起 [batch_size, num_category, height, width]
        ch_aggregated_x = torch.cat(weighted_channels, dim=1)
        
        # 聚合图强度归一化
        # x_min = ch_aggregated_x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0].min(dim=-3, keepdim=True)[0] # 计算最小值
        # x_max = ch_aggregated_x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=-3, keepdim=True)[0] # 和最大值
        # ch_aggregated_x = (ch_aggregated_x - x_min) / (x_max - x_min + 1e-6) # 线性归一化到 [0, 1]
        # x = 2 * x_norm_01 - 1 # 线性归一化到 [-1, 1]
        # if class_msk is not None:
        #     ch_agg_min = ch_aggregated_x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0].min(dim=-3, keepdim=True)[0]
        #     # torch.where(condition, value_if_true, value_if_false)
        #     ch_aggregated_x = torch.where(class_msk.unsqueeze(-1).unsqueeze(-1)==1, ch_aggregated_x,  ch_agg_min)
        x = ch_aggregated_x.view(-1, 1, h, w) # 重塑为 [batch_size*num_category, 1, height, width]
        
        if vis_dirs:
            for i, vd in enumerate(vis_dirs):
                self.visualize_features(ch_aggregated_x[i, ...], vd)
        
        encode_outs =[] 
        decode_outs = []
        side_outs = []
        
        for m in self.encode_modules[:-1]: # skip the last bridge module named as enc6
            x = m(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
            encode_outs.append(x)
        # encode_outs: [enc1, enc2, enc3, enc4, enc5]
            
        x = self.encode_modules[-1](x)
        decode_outs.append(x)
        # decode_outs: [enc6 ]

        for m, x2 in zip(self.decode_modules, reversed(encode_outs)):
            x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=False)
            x = m(torch.cat([x, x2], dim=1))
            decode_outs.append(x)

        for m, x in zip(self.side_modules, decode_outs):
            x = F.interpolate(m(x), size=(h, w), mode="bilinear", align_corners=False) # x.shape is [b*n_c, out_ch , h,w]
            side_outs.insert(0, x) 
            
        x = self.out_conv(torch.cat(side_outs, dim=1)) # [batch_size*num_category, out_ch, height, width]
        if self.out_ch ==  1:
            x = x.view(b, self.n_classes, h, w) # x.shape is [b, n_c*out_ch, h,w] out_ch = 1

            side_outs = [so.view(b, self.n_classes, h, w) for so in side_outs]
        if self.training:
            # do not use torch.sigmoid for amp safe 二分类任务中,我们通常使用sigmoid函数作为输出层的激活函数,在多分类任务中,我们通常使用softmax函数作为输出层的激活函数,
            return [x] + side_outs
        else:
            return x
     
def save_grad(name):
    def hook(grad):
        print(f"Gradient for {name}: {grad}")
    return hook
        
def coch_u2net_full(ch_cl_mapping,
                    in_ch=1, n_classes: int=11,
                    out_ch: int=1,
                    ) -> Coch_U2NET:
    cfg = {
        # [height, in_ch, mid_ch, out_ch, RSU4F, side]
        "encode": [
            [7, in_ch, 8, 16, False, False], # enc1
            [6, 16, 8, 32, False, False],    # enc2
            [5, 32, 16, 64, False, False],   # enc3
            [4, 64, 32, 128, False, False],  # enc4
            # [4, 128, 64, 128, True, False],   # enc5
            [4, 128, 64, 128, True, True],    # enc6 as bridge
        ],
        # [height, in_ch, mid_ch, out_ch, RSU4F, side]
        "decode": [
            # [4, 256, 64, 128, True, True],    # dec5
            [4, 256, 32, 64, False, True],   # dec4
            [5, 128, 16, 32, False, True],   # dec3
            [6, 64, 8, 16, False, True],   # dec2
            [7, 32, 4, 16, False, True],    # dec1
        ]
    }
    return Coch_U2NET(cfg, n_classes, ch_cl_mapping, out_ch=out_ch)

def coch_u2net_lite(ch_cl_mapping,
                    in_ch=1, n_classes: int=11,
                    out_ch: int=1,) -> Coch_U2NET:
    cfg = {
        # [height, in_ch, mid_ch, out_ch, RSU4F, side]
        "encode": [[7, in_ch, 8, 32, False, False],  # En1
                   [6, 32, 8, 32, False, False],  # En2
                   [5, 32, 8, 32, False, False],  # En3
                   [4, 32, 8, 32, False, False],  # En4
                #    [4, 32, 8, 32, True, False],  # En5
                   [4, 32, 8, 32, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 64, 8, 32, True, True],  # De5
                #    [4, 64, 8, 32, False, True],  # De4
                   [5, 64, 8, 32, False, True],  # De3
                   [6, 64, 8, 32, False, True],  # De2
                   [7, 64, 8, 32, False, True]]  # De1
    }
    return Coch_U2NET(cfg, n_classes, ch_cl_mapping, out_ch=out_ch)

def convert_onnx(m, save_path):
    m.eval()
    x = torch.randn(1, 60, 256, 256, requires_grad=True)
    
    torch.onnx.export(m, 
                      x, 
                      save_path, 
                      export_params=True,
                      opset_version=11, )
