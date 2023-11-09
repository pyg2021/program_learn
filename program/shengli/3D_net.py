import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from DataLoad import DataLoad
from Model import net
start=time.time()

##data_prepare





# sio.loadmat()
# seed=20
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
device="cuda"
# v=sio.loadmat("/home/pengyaoguang/data/3D_net_result/floed_v0.mat")["v"]
# y=v.reshape(1,1,v.shape[0],v.shape[1],v.shape[2])


# R=sio.loadmat("/home/pengyaoguang/data/3D_net_result/RTM_easy1022.mat")["RTM"][10:110,10:110,10:110]
# plt.figure()
# plt.imshow(R[:,50,:].T)
# plt.colorbar()
# plt.savefig("/home/pengyaoguang/data/3D_net_result/RTM.png")
# v_smooth=1/gaussian_filter(1/v,sigma=10)
# v_smooth=v_smooth.reshape(1,1,R.shape[0],R.shape[1],R.shape[2])
# x=np.zeros((1,2,R.shape[0],R.shape[1],R.shape[2]))
# R=R.reshape(1,1,R.shape[0],R.shape[1],R.shape[2])
# x[:,0,:,:,:]=R
# x[:,1,:,:,:]=v_smooth

x,y=DataLoad(12)
x=torch.from_numpy(x).float().to(device)
y=torch.from_numpy(y).float().to(device)



model=net(2,1).to(device)
model=nn.parallel.DataParallel(model)


epoch=40000
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-2)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=2000,gamma=0.8)
loss_1=torch.nn.L1Loss()
plt.figure()
plt.imshow(y.cpu().detach()[0,0,50,:,:].T)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/3D_net_result/v_real.png")
plt.close()
# plt.figure()
# plt.imshow(v_smooth[0,0,50,:,:].T)
# plt.colorbar()
# plt.savefig("/home/pengyaoguang/data/3D_net_result/v_smooth.png")
# plt.close()
for i in range(epoch):
    optimizer.zero_grad()
    y_1=model(x)
    loss=loss_1(y_1,y)+1*loss_1(torch.clamp(y_1,1.5,8),y_1)
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
        torch.save(model.state_dict(),"/home/pengyaoguang/data/3D_net_model/modeltest.pkl")


##2023年11月6日23:17
##2023年11月7日10:03
##2023年11月8日13:07
