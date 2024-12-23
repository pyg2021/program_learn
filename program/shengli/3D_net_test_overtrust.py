import torch
from Model3D_unt import net
import torch.nn as nn
import scipy.io as sio
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from DataLoad import DataLoad
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
device="cuda"
model=net(2,1,True,True).to(device)
model=nn.parallel.DataParallel(model)
model.load_state_dict(torch.load("/home/pengyaoguang/data/3D_net_model/modeltest6_4.pkl"))



m=0
##data_prepare
k=10102
n=50
R=sio.loadmat("/home/pengyaoguang/data/3D_RTM/overtrust_RTM.mat")["RTM"][20:120,20:120,20:120]

vmax=np.max(R)
plt.figure()
plt.imshow(R[n].T/vmax,cmap="gray")
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_result/RTM_test{}.png".format(m))


R1=R.reshape(1,1,R.shape[0],R.shape[1],R.shape[2])
label=sio.loadmat("/home/pengyaoguang/data/3D_v/Overthrust_vel0.mat")["v"]/1000


plt.figure()
plt.imshow(label[n].T,vmin=2,vmax=5)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_result/v_real_test{}.png".format(m))


label1=label.reshape(1,1,label.shape[0],label.shape[1],label.shape[2])
label_smooth=1/gaussian_filter(1/label,sigma=10)

plt.figure()
plt.imshow(label_smooth[n].T,vmin=2,vmax=5)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_result/v_smooth_test{}.png".format(m))

label_smooth1=label_smooth.reshape(1,1,label_smooth.shape[0],label_smooth.shape[1],label_smooth.shape[2])
x=np.zeros((1,2,100,100,100))
x[:,0]=R1
x[:,1]=label_smooth1



##test
x=torch.from_numpy(x).float().to(device)
label1=torch.from_numpy(label1).float().to(device)

# device="cuda"
# x,y=DataLoad(2)
# x=torch.from_numpy(x).float().to(device)
# y=torch.from_numpy(y).float().to(device)
# label1=y

loss_1=torch.nn.L1Loss()
y_1=model(x)
loss=loss_1(y_1,label1)
print(loss)

plt.figure()
plt.imshow(y_1.detach().cpu()[0,0,n].T,vmin=2,vmax=5)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_result/v_updete_test{}.png".format(m))


plt.figure()
plt.imshow(y_1.detach().cpu()[0,0,n].T-torch.from_numpy(label[n].T))
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_result/v_error{}.png".format(m))
