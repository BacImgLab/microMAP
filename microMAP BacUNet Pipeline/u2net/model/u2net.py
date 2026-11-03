'''
Author: Hai-rf 1224483724@qq.com
Date: 2024-11-08 21:48:56
LastEditTime: 2024-11-12 01:59:03
LastEditors: Hai-rf 1224483724@qq.com
Description: 
'''
from typing import Union, List
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        
        #self.bn = nn.BatchNorm2d(out_ch)
        self.bn = nn.GroupNorm(8, out_ch)
        #print("here")
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
    
class U2NET(nn.Module):
    def __init__(self, cfg: dict, n_classes: int):
        assert "encode" in cfg, "cfg should contain 'encode' key"
        assert "decode" in cfg, "cfg should contain 'decode' key"
        self.encode_num = len(cfg["encode"])
        
        super().__init__()
        
        self.n_classes = n_classes

        encode_list = []
        decode_list = []
        side_list = []
        
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            encode_list.append( RSU4F(*c[1:4]) if c[4] else RSU(*c[:4]) )
            if c[5]:
                side_list.append(nn.Conv2d(c[3], n_classes, kernel_size=3, padding=1))

        for c in cfg["decode"]:
            decode_list.append( RSU4F(*c[1:4]) if c[4] else RSU(*c[:4]) )
            if c[5]:
                side_list.append(nn.Conv2d(c[3], n_classes, kernel_size=3, padding=1))
                
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
        self.out_conv = nn.Conv2d(self.encode_num * n_classes, n_classes, kernel_size=1)
        init.kaiming_normal_(self.out_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.out_conv.bias is not None:
            init.zeros_(self.out_conv.bias)
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w, = x.shape
        
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
            x = F.interpolate(m(x), size=(h, w), mode="bilinear", align_corners=False)
            side_outs.insert(0, x) # x.shape is [b,n_classes, h,w]
            
        x = self.out_conv(torch.cat(side_outs, dim=1))
        # x = self.out_conv1(torch.cat(side_outs, dim=1))
        # x = self.relu(x)
        # x = self.out_conv2(x)

        if self.training:
            # do not use torch.sigmoid for amp safe 二分类任务中,我们通常使用sigmoid函数作为输出层的激活函数,在多分类任务中,我们通常使用softmax函数作为输出层的激活函数,
            return [x] + side_outs
        else:
            return x
            #return [x] + side_outs
     
def save_grad(name):
    def hook(grad):
        print(f"Gradient for {name}: {grad}")
    return hook
    

class MLU2NET(nn.Module):
    def __init__(self, cfg: dict, n_classes: int, ):
        assert "encode" in cfg, "cfg should contain 'encode' key"
        assert "decode" in cfg, "cfg should contain 'decode' key"
        self.encode_num = len(cfg["encode"])
        
        super().__init__()
        
        self.n_classes = n_classes

        encode_list = []
        decode_list = []
        side_list = []
        
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            encode_list.append( RSU4F(*c[1:4]) if c[4] else RSU(*c[:4]) )
            if c[5]:
                side_list.append(nn.Conv2d(c[3], 1, kernel_size=3, padding=1))

        for c in cfg["decode"]:
            decode_list.append( RSU4F(*c[1:4]) if c[4] else RSU(*c[:4]) )
            if c[5]:
                side_list.append(nn.Conv2d(c[3], 1, kernel_size=3, padding=1))
                
        self.encode_modules = nn.ModuleList(encode_list) # [enc1, enc2, enc3, enc4, enc5, enc6 as bridge]
        
        self.decode_modules_group = nn.ModuleList(
            [nn.ModuleList(copy.deepcopy(decode_list)) for _ in range(n_classes)]
        ) # [[dec5, dec4, dec3, dec2, dec1] * k]
        # self.side_modules = nn.ModuleList(side_list)
        self.side_modules_group = nn.ModuleList(
            [nn.ModuleList(copy.deepcopy(side_list)) for _ in range(n_classes)]
        )
        
       
        self.out_conv = nn.Conv2d(self.encode_num * 1, 1, kernel_size=1)
        init.kaiming_normal_(self.out_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.out_conv.bias is not None:
            init.zeros_(self.out_conv.bias)
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w, = x.shape
        
        encode_outs =[] 
        decode_outs = [[] for _ in range(self.n_classes)]
        side_outs = [[] for _ in range(self.n_classes)]
        ch_outs = []
        
        for m in self.encode_modules[:-1]: # skip the last bridge module named as enc6
            x = m(x)
           
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
            encode_outs.append(x)
        # x.register_hook(save_grad("en_out"))
        # encode_outs: [enc1, enc2, enc3, enc4, enc5]
            
        x = self.encode_modules[-1](x)
        [de_out.append(x) for de_out in decode_outs]
        # decode_outs: [enc6 ]

        for decode_modules, de_out in zip(self.decode_modules_group, decode_outs):
            x = de_out[0]
            for m, x2 in zip(decode_modules, reversed(encode_outs)):
                x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=False)
                x = m(torch.cat([x, x2], dim=1))
                de_out.append(x)
        for side_modules, de_out, si_out in zip(self.side_modules_group, decode_outs, side_outs):
            for m, x in zip(side_modules, de_out):
                x = F.interpolate(m(x), size=(h, w), mode="bilinear", align_corners=False)
                si_out.insert(0, x)
            # x.register_hook(save_grad("side_module"))
            
            ch_out = self.out_conv(torch.cat(si_out, dim=1))
            ch_outs.append(ch_out)
        # ch_out.register_hook(save_grad("side_ch_out"))
            
        # x = self.out_conv1(torch.cat(side_outs, dim=1))
        # x = self.relu(x)
        # x = self.out_conv2(x)
        
        if self.training:
            side_cat_outs = []
            for i in range(len(side_outs[0])):
                side_cat_outs.append(torch.cat([side_out[i] for side_out in side_outs], dim=1))
            # for side_out in side_cat_outs:
            #     print("side_cat", side_out.grad)
            # side_out = side_cat_outs[0]
            # side_out.register_hook(save_grad("side_cat"))
            x = torch.cat(ch_outs, dim=1) 
            # do not use torch.sigmoid for amp safe 二分类任务中,我们通常使用sigmoid函数作为输出层的激活函数,在多分类任务中,我们通常使用softmax函数作为输出层的激活函数,
            return [x] + side_cat_outs
        else:
            x = torch.cat(ch_outs, dim=1) 
            return x
        
def u2net_full(img_ch=60, n_classes: int=12) -> U2NET:
    cfg = {
        # [height, in_ch, mid_ch, out_ch, RSU4F, side]
        "encode": [
            [7, img_ch, 32, 64, False, False], # enc1
            [6, 64, 32, 128, False, False],    # enc2
            [5, 128, 64, 256, False, False],   # enc3
            [4, 256, 128, 512, False, False],  # enc4
            [4, 512, 256, 512, True, False],   # enc5
            [4, 512, 256, 512, True, True],    # enc6 as bridge
        ],
        # [height, in_ch, mid_ch, out_ch, RSU4F, side]
        "decode": [
            [4, 1024, 256, 512, True, True],    # dec5
            [4, 1024, 128, 256, False, True],   # dec4
            [5, 512, 64, 128, False, True],   # dec3
            [6, 256, 32, 64, False, True],   # dec2
            [7, 128, 16, 64, False, True],    # dec1
        ]
    }
    return U2NET(cfg, n_classes)

def ch2cl_u2net_full(ch_cl_mapping,
                    img_ch=60, n_classes: int=11,
                     
                     ) -> U2NET:
    in_ch = n_classes
    cfg = {
        # [height, in_ch, mid_ch, out_ch, RSU4F, side]
        "encode": [
            [7, in_ch, 32, 64, False, False], # enc1
            [6, 64, 32, 128, False, False],    # enc2
            [5, 128, 64, 256, False, False],   # enc3
            [4, 256, 128, 512, False, False],  # enc4
            [4, 512, 256, 512, True, False],   # enc5
            [4, 512, 256, 512, True, True],    # enc6 as bridge
        ],
        # [height, in_ch, mid_ch, out_ch, RSU4F, side]
        "decode": [
            [4, 1024, 256, 512, True, True],    # dec5
            [4, 1024, 128, 256, False, True],   # dec4
            [5, 512, 64, 128, False, True],   # dec3
            [6, 256, 32, 64, False, True],   # dec2
            [7, 128, 16, 64, False, True],    # dec1
        ]
    }
    return U2NET(cfg, n_classes)

def mlu2net_full(img_ch=60, n_classes: int=11) -> U2NET:
    cfg = {
        # [height, in_ch, mid_ch, out_ch, RSU4F, side]
        "encode": [
            [7, img_ch, 32, 64, False, False], # enc1
            [6, 64, 32, 128, False, False],    # enc2
            [5, 128, 64, 256, False, False],   # enc3
            [4, 256, 128, 512, False, False],  # enc4
            [4, 512, 256, 512, True, False],   # enc5
            [4, 512, 256, 512, True, True],    # enc6 as bridge
        ],
        # [height, in_ch, mid_ch, out_ch, RSU4F, side]
        "decode": [
            [4, 1024, 256, 512, True, True],    # dec5
            [4, 1024, 128, 256, False, True],   # dec4
            [5, 512, 64, 128, False, True],   # dec3
            [6, 256, 32, 64, False, True],   # dec2
            [7, 128, 16, 64, False, True],    # dec1
        ]
    }
    return MLU2NET(cfg, n_classes)

def u2net_lite(img_ch=60, n_classes: int=11) -> U2NET:
    cfg = {
        # [height, in_ch, mid_ch, out_ch, RSU4F, side]
        "encode": [[7, img_ch, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }
    return U2NET(cfg, n_classes)

def convert_onnx(m, save_path):
    m.eval()
    x = torch.randn(1, 60, 256, 256, requires_grad=True)
    
    torch.onnx.export(m, 
                      x, 
                      save_path, 
                      export_params=True,
                      opset_version=11, )

        