import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False):
        super(DoubleConv, self).__init__()
        channels = out_channels // 2
        if in_channels > out_channels:
            channels = in_channels // 2

        layers = [
            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        ]
        if batch_normal:
            layers.insert(1, nn.InstanceNorm3d(channels))
            layers.insert(len(layers) - 1, nn.BatchNorm3d(out_channels))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False):
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, batch_normal)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False, bilinear=True):
        super(UpSampling, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + in_channels // 2, out_channels, batch_normal)

    def forward(self, inputs1, inputs2):
        inputs1 = self.up(inputs1)
        pad1=inputs2.size()[-3]-inputs1.size()[-3]
        pad2=inputs2.size()[-2]-inputs1.size()[-2]
        pad3=inputs2.size()[-1]-inputs1.size()[-1]
        
        inputs1=F.pad(inputs1,[pad3//2,pad3+1//2,pad2//2,pad2+1//2,pad1//2,pad1+1//2])
        outputs = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.conv(outputs)
        return outputs

class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super(LastConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1 )

    def forward(self, x):
        return self.conv(x)

class net(nn.Module):
    def __init__(self, in_channels, num_classes=2, batch_normal=False, bilinear=True):
        super(net, self).__init__()
        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.bilinear = bilinear
        fiter=[64,128,256,512]
        
        self.inputs = DoubleConv(in_channels, fiter[0], self.batch_normal)
        self.down_1 = DownSampling(fiter[0], fiter[1], self.batch_normal)
        self.down_2 = DownSampling(fiter[1], fiter[2], self.batch_normal)
        self.down_3 = DownSampling(fiter[2], fiter[3], self.batch_normal)

        self.up_1 = UpSampling(fiter[3], fiter[2], self.batch_normal, self.bilinear)
        self.up_2 = UpSampling(fiter[2], fiter[1], self.batch_normal, self.bilinear)
        self.up_3 = UpSampling(fiter[1], fiter[0], self.batch_normal, self.bilinear)
        self.outputs = LastConv(64, num_classes)

    def forward(self, x):
        x1 = self.inputs(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)

        x5 = self.up_1(x4, x3)
        x6 = self.up_2(x5, x2)
        x7 = self.up_3(x6, x1)
        x = self.outputs(x7)

        return x


# device='cuda:0'
# model=net(2,1).to(device)
# # model=nn.parallel.DataParallel(model)
# x=np.ones((1,2,100,100,100))
# x=torch.from_numpy(x).float().to(device)
# y=model(x)
# print(y.shape)