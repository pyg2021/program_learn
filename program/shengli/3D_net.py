import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
start=time.time()
class conv_net(nn.Module):
    def __init__(self,in_channels, out_channels, is_batchnorm=True):
        super(conv_net,self).__init__()
        self.conv0=nn.Sequential(nn.Conv3d(in_channels,in_channels,3,1,1),
                                 nn.BatchNorm3d(in_channels),
                                 nn.ReLU(inplace=True),)
        self.conv1=nn.Sequential(nn.Conv3d(in_channels,out_channels,3,1,1),
                                 nn.BatchNorm3d(out_channels),
                                 nn.ReLU(inplace=True),)
                                #  nn.MaxPool3d(2,2,ceil_mode=True))
        self.conv2=nn.Sequential(nn.Conv3d(out_channels,out_channels,3,1,1),
                                 nn.BatchNorm3d(out_channels),
                                 nn.ReLU(inplace=True),)
    def forward(self,input):
        x=self.conv0(input)
        x=self.conv1(x)
        return self.conv2(x)

class net(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, is_batchnorm=True):
        super(net, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.is_batchnorm=is_batchnorm
        # self.conv3d=nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        # self.batch=nn.BatchNorm3d(out_channels)
        # self.pool=nn.MaxPool3d(2,2)
        n=6
        self.conv1=conv_net(self.in_channels,n)
        self.conv2=conv_net(n,n)
        self.conv3= conv_net(n,self.out_channels)
        self.conv4=nn.Sequential(nn.Conv3d(out_channels,out_channels,3,1,1),
                                 nn.BatchNorm3d(out_channels),
                                 nn.ReLU(inplace=True),
                                 )
        self.conv5=nn.Sequential(nn.Conv3d(out_channels,out_channels,3,1,1),
                                 nn.ReLU(inplace=True),)
    def forward(self,input):
        x=self.conv1(input)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        return x

# sio.loadmat()
seed=20
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
device="cuda:2"
v=sio.loadmat("/home/pengyaoguang/data/3D_net_result/floed_v0.mat")["v"]
y=v.reshape(1,1,v.shape[0],v.shape[1],v.shape[2])
y=torch.from_numpy(y).float().to(device)

R=sio.loadmat("/home/pengyaoguang/data/3D_net_result/RTM_easy1022.mat")["RTM"][10:110,10:110,10:110]
plt.figure()
plt.imshow(R[:,50,:].T)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_result/RTM.png")
v_smooth=1/gaussian_filter(1/v,sigma=3)
v_smooth=v_smooth.reshape(1,1,R.shape[0],R.shape[1],R.shape[2])
x=np.zeros((1,1,R.shape[0],R.shape[1],R.shape[2]))
R=R.reshape(1,1,R.shape[0],R.shape[1],R.shape[2])
x[:,0,:,:,:]=R
# x[:,1,:,:,:]=v_smooth
x=torch.from_numpy(x).float().to(device)
model=net(1,1).to("cuda:2")
epoch=20000
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-2)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=2000,gamma=0.8)
loss_1=torch.nn.L1Loss()
plt.figure()
plt.imshow(y.cpu().detach()[0,0,50,:,:].T)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_result/v_real.png")
plt.close()
plt.figure()
plt.imshow(v_smooth[0,0,50,:,:].T)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_result/v_smooth.png")
plt.close()
for i in range(epoch):
    optimizer.zero_grad()
    y_1=model(x)
    loss=loss_1(y_1,y)+1*loss_1(torch.clamp(y_1,1.8,7),y_1)
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(i,loss)
    if i%50==0:
        print(time.time()-start,"s")
        plt.figure()
        plt.imshow(model(x).cpu().detach()[0,0,50,:,:].T)
        plt.colorbar()
        plt.savefig("/home/pengyaoguang/data/3D_net_result/v_updata.png")
        plt.close()
        sio.savemat("/home/pengyaoguang/data/3D_net_result/v_updata.mat",{"v":model(x).cpu().detach()[0,0]})
##2023年11月6日23:17
##2023年11月7日10:03