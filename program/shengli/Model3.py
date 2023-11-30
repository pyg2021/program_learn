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
        self.conv1=nn.Sequential(nn.Conv3d(in_channels,out_channels,3,1,1),
                                 nn.BatchNorm3d(out_channels),
                                 nn.ReLU(inplace=True),
                                 )
    def forward(self,input):
        x=self.conv1(input)
        return x

class net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, is_batchnorm=True):
        super(net, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.is_batchnorm=is_batchnorm
        # self.conv3d=nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        # self.batch=nn.BatchNorm3d(out_channels)
        # self.pool=nn.MaxPool3d(2,2)
        filter=[32,64,128,256,512,1024]
        self.conv1=conv_net(self.in_channels,filter[0])
        self.conv2=conv_net(filter[0],filter[1])
        self.down=nn.Sequential(nn.Conv3d(filter[1],filter[2],3,1,1),
                                 nn.BatchNorm3d(filter[2]),
                                 nn.ReLU(inplace=True),
                                 )
        self.conv3= conv_net(filter[2],filter[1])

        self.conv4= conv_net(filter[1],filter[0])
        self.conv5= conv_net(filter[0],in_channels)
        self.conv6=nn.Sequential(nn.Conv3d(in_channels,out_channels,3,1,1),
                                 nn.ReLU(inplace=True),
                                 )
        self.conv7=nn.Sequential(nn.Conv3d(out_channels,out_channels,3,1,1),
                                 nn.ReLU(inplace=True),)
    def forward(self,input):
        x=self.conv1(input)
        x=self.conv2(x)
        x=self.down(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.conv6(x)
        x=self.conv7(x)
        return x

# device='cuda:1'
# model=net(2,1).to(device)
# x=np.zeros((1,2,100,100,100))
# x=torch.from_numpy(x).float().to(device)
# print(model(x).shape)

