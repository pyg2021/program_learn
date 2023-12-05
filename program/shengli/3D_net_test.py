import torch
from Model3 import net
import torch.nn as nn
import scipy.io as sio
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from DataLoad import DataLoad
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
device="cuda"
model=net(2,1).to(device)
model=nn.parallel.DataParallel(model)
model.load_state_dict(torch.load("/home/pengyaoguang/data/3D_net_model/modeltest5.pkl"))




##data_prepare
k=10006
n=50
R=sio.loadmat("/home/pengyaoguang/data/3D_RTM/RTM{}".format(k))["RTM"][20:120,20:120,20:120]

plt.figure()
plt.imshow(R[n].T,cmap="gray")
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_result/RTM_test.png")


R1=R.reshape(1,1,R.shape[0],R.shape[1],R.shape[2])
label=sio.loadmat("/home/pengyaoguang/data/3D_v_model/v{}".format(k))["v"]


plt.figure()
plt.imshow(label[n].T)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_result/v_real_test.png")


label1=label.reshape(1,1,label.shape[0],label.shape[1],label.shape[2])
label_smooth=1/gaussian_filter(1/label,sigma=10)

plt.figure()
plt.imshow(label_smooth[n].T)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_result/v_smooth_test.png")

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
plt.imshow(y_1.detach().cpu()[0,0,n].T)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_result/v_updete_test.png")


plt.figure()
plt.imshow(y_1.detach().cpu()[0,0,n].T-torch.from_numpy(label[n].T))
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_result/v_error.png")
