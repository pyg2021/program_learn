import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from DataLoad import DataLoad
from Model3 import net
import os 
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
start=time.time()

##data_prepare

device="cuda"
x,y=DataLoad(20)
x=torch.from_numpy(x).float().to(device)
y=torch.from_numpy(y).float().to(device)



model=net(2,1).to(device)
model=nn.parallel.DataParallel(model)
# model.load_state_dict(torch.load("/home/pengyaoguang/data/3D_net_model/modeltest4_2.pkl"))

epoch=20000
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-2)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1500,gamma=0.8)
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
loss_all=[]
for i in range(epoch):
    optimizer.zero_grad()
    y_1=model(x)
    loss=loss_1(y_1,y)+2*loss_1(torch.clamp(y_1,1.5,8),y_1)
    # loss=loss_1(y_1,y)+1*loss_1(torch.clamp(y_1,1.5,8),y_1)+loss_1(y_1[:,:,50,50,:],y[:,:,50,50,:])+loss_1(y_1[:,:,10,50,:],y[:,:,10,50,:])+loss_1(y_1[:,:,50,10,:],y[:,:,50,10,:])
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(i,loss)
    loss_all.append(loss.detach().cpu().item())
    if i%50==0:
        print(time.time()-start,"s")
        plt.figure()
        plt.imshow(model(x).cpu().detach()[0,0,50,:,:].T)
        plt.colorbar()
        plt.savefig("/home/pengyaoguang/data/3D_net_result/v_updata4.png")
        plt.close()
        sio.savemat("/home/pengyaoguang/data/3D_net_result/v_updata4.mat",{"v":model(x).cpu().detach()[0,0]})
        torch.save(model.state_dict(),"/home/pengyaoguang/data/3D_net_model/modeltest4_2.pkl")

        plt.figure()
        plt.plot(range(len(loss_all)),loss_all,label="train")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("/home/pengyaoguang/data/3D_net_result/history4.png")
        plt.close()

##2023年11月28日10点45
