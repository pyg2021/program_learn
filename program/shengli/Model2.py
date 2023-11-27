import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
class conv_net(nn.Module):
    def __init__(self,in_channels, out_channels, is_batchnorm=True):
        super(conv_net,self).__init__()
        self.conv0=nn.Sequential(nn.Conv3d(in_channels,in_channels,3,1,1),
                                 nn.BatchNorm3d(in_channels),
                                 )
        self.conv1=nn.Sequential(nn.Conv3d(in_channels,out_channels,3,1,1),
                                 nn.BatchNorm3d(out_channels),
                                 )
                                #  nn.MaxPool3d(2,(1,1,2),ceil_mode=True))
        self.conv2=nn.Sequential(nn.Conv3d(out_channels,out_channels,3,1,1),
                                 nn.BatchNorm3d(out_channels),
                                 nn.ReLU(inplace=True),)
    def forward(self,input):
        x=self.conv0(input)
        x=self.conv1(x)
        x=self.conv2(x)
        return x

class net(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, is_batchnorm=True):
        super(net, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.is_batchnorm=is_batchnorm
        # self.conv3d=nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        # self.batch=nn.BatchNorm3d(out_channels)
        # self.pool=nn.MaxPool3d(2,2)
        n=200
        self.conv1=conv_net(self.in_channels,in_channels)
        self.conv2=conv_net(in_channels,n)
        self.down=nn.Sequential(nn.Conv3d(n,n,3,(1,1,2),(1,1,0)),
                                nn.Conv3d(n,n,(5,5,26),1,(2,2,0)),
                                 nn.BatchNorm3d(n),
                                 nn.ReLU(inplace=True),
                                 )
        self.conv3= conv_net(n,self.out_channels)
        self.conv4=nn.Sequential(nn.Conv3d(out_channels,out_channels,3,1,1),
                                 nn.ReLU(inplace=True),
                                 )
        self.conv5=nn.Sequential(nn.Conv3d(out_channels,out_channels,3,1,1),
                                 nn.ReLU(inplace=True),)
    def forward(self,input):
        x=self.conv1(input)
        x=self.conv2(x)
        x=self.down(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        return x

# device='cuda'
# model=net(25,1).to(device)
# x=np.zeros((25,25,100,100,252))
# x=torch.from_numpy(x).float().to(device)
# print(model(x).shape)

