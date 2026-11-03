from typing import Union, List
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        encode_list = []
        decode_list = []
        
        encode_list.append(DoubleConv(n_channels, 64))
        encode_list.append(Down(64, 128))
        encode_list.append(Down(128, 256))
        encode_list.append(Down(256, 512))
        factor = 2 if bilinear else 1
        encode_list.append(Down(512, 1024 // factor))
        
        decode_list.append(Up(1024, 512 // factor, bilinear))
        decode_list.append(Up(512, 256 // factor, bilinear))
        decode_list.append(Up(256, 128 // factor, bilinear))
        decode_list.append(Up(128, 64, bilinear))
        
        self.encode_modules = nn.ModuleList(encode_list) # [inc, down1, down2, down3, down4]
        self.decode_modules = nn.ModuleList(decode_list) # [up1, up2, up3, up4]
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        encode_outs = []
        for m in self.encode_modules[:-1]:
            x = m(x)
            encode_outs.append(x)
            
        x = self.encode_modules[-1](x)
    
        for m, x2 in zip(self.decode_modules, reversed(encode_outs)):
            x = m(x, x2)

        logits = self.out_conv(x)
        if self.training:
            return [logits]
        else:
            return logits
            