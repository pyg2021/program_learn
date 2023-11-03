import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
start=time.time()
class conv_net(nn.Module):
    def __init__(self,in_channels, out_channels, is_batchnorm=True):
        super(conv_net,self).__init__()
        self.conv1=nn.Sequential(nn.Conv3d(in_channels,out_channels,3,1,1),
                                 nn.BatchNorm3d(out_channels),
                                 nn.ReLU(inplace=True),)
                                #  nn.MaxPool3d(2,2,ceil_mode=True))
        self.conv2=nn.Sequential(nn.Conv3d(out_channels,out_channels,3,1,1),
                                 nn.BatchNorm3d(out_channels),
                                 nn.ReLU(inplace=True),)
    def forward(self,input):
        x=self.conv1(input)
        return self.conv2(x)

class net(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, is_batchnorm=True):
        super(net, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.is_batchnorm=is_batchnorm
        self.conv3d=nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        self.batch=nn.BatchNorm3d(out_channels)
        self.pool=nn.MaxPool3d(2,2)
        self.conv1=conv_net(self.in_channels,self.out_channels)
        self.conv2=conv_net(self.out_channels,self.out_channels)
    def forward(self,input):
        print(input.shape)
        # x=self.conv3d(input)
        # print(x.shape)
        # x=self.batch(x)
        # print(x.shape)
        # x=self.pool(x)
        # print(x.shape)
        x=self.conv1(input)
        x=self.conv2(input)
        return x

x=torch.ones(100,1, 100,150,200).to("cuda")
model=net(1,1).to("cuda")
# print(net())
y=model(x)
# print(y)
print(time.time()-start,"s")
