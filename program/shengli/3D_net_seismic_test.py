import torch
from Model2 import net
import torch.nn as nn
import scipy.io as sio
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
device="cuda"
model=net(25,1).to(device)
model=nn.parallel.DataParallel(model)
model.load_state_dict(torch.load("/home/pengyaoguang/data/3D_net_seismic_model/modeltest1.pkl"))



size=40
##data_prepare
x=np.zeros((1,25,100,100,252))
y=np.zeros((1,1,100,100,100))
n=50
for k in range(size,size+1):
    for i in range(25):
        R=sio.loadmat("/home/pengyaoguang/data/3D_seismic_data/seismic{}_{}.mat".format(k,i))["seismic_data"]
        R1=R.reshape(1,1,R.shape[0],R.shape[1],R.shape[2])
        x[0,i]=R1
    label=sio.loadmat("/home/pengyaoguang/data/3D_v_model/v{}".format(k))["v"]
    label1=label.reshape(1,1,label.shape[0],label.shape[1],label.shape[2])
    label_smooth=1/gaussian_filter(1/label,sigma=10)
    label_smooth1=label_smooth.reshape(1,1,label_smooth.shape[0],label_smooth.shape[1],label_smooth.shape[2])
    # x[k,-1]=label_smooth1
    y[0,0]=label1


##test
x=torch.from_numpy(x).float().to(device)
label1=torch.from_numpy(y).float().to(device)


loss_1=torch.nn.L1Loss()
y_1=model(x)
loss=loss_1(y_1,label1)
print(loss)

plt.figure()
plt.imshow(y_1.detach().cpu()[0,0,n].T)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_seismic/v_updete_test.png")


plt.figure()
plt.imshow(y_1.detach().cpu()[0,0,n].T-torch.from_numpy(label[n].T))
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_seismic/v_error.png")
