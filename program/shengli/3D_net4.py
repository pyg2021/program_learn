import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
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
BatchSize=20
device="cuda"
x,y=DataLoad(60)
train_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
train_loader = data_utils.DataLoader(train_data,batch_size=BatchSize,shuffle=True)
# x=torch.from_numpy(x).float().to(device)
# y=torch.from_numpy(y).float().to(device)



model=net(2,1).to(device)
model=nn.parallel.DataParallel(model)
# model.load_state_dict(torch.load("/home/pengyaoguang/data/3D_net_model/modeltest4_2.pkl"))

epoch=20000
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-2)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1500,gamma=0.8)
loss_1=torch.nn.L1Loss()
# plt.figure()
# plt.imshow(y.cpu().detach()[0,0,50,:,:].T)
# plt.colorbar()
# plt.savefig("/home/pengyaoguang/data/3D_net_result/v_real.png")
# plt.close()

loss_all=[]

for epoch_i in range(epoch):
    epoch_loss=0
    for i,(x,y) in enumerate(train_loader):
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        y_1=model(x)
        loss=loss_1(y_1,y)+2*loss_1(torch.clamp(y_1,1.5,8),y_1)
        # loss=loss_1(y_1,y)+1*loss_1(torch.clamp(y_1,1.5,8),y_1)+loss_1(y_1[:,:,50,50,:],y[:,:,50,50,:])+loss_1(y_1[:,:,10,50,:],y[:,:,10,50,:])+loss_1(y_1[:,:,50,10,:],y[:,:,50,10,:])
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss+=loss.detach().cpu().item()
    loss_all.append(epoch_loss)
    print(epoch_i,epoch_loss)
    if epoch_i%50==0:
        print(time.time()-start,"s")
        plt.figure()
        plt.imshow(model(x).cpu().detach()[0,0,50,:,:].T)
        plt.colorbar()
        plt.savefig("/home/pengyaoguang/data/3D_net_result/v_updata4.png")
        plt.close()
        sio.savemat("/home/pengyaoguang/data/3D_net_result/v_updata4.mat",{"v":model(x).cpu().detach()[0,0]})
        torch.save(model.state_dict(),"/home/pengyaoguang/data/3D_net_model/modeltest4_4.pkl")

        plt.figure()
        plt.imshow(y.cpu().detach()[0,0,50,:,:].T)
        plt.colorbar()
        plt.savefig("/home/pengyaoguang/data/3D_net_result/v_real.png")
        plt.close()

        plt.figure()
        plt.plot(range(len(loss_all)),loss_all,label="train")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("/home/pengyaoguang/data/3D_net_result/history4.png")
        plt.close()

##2023年11月28日10点45
##2023年12月1日10点32 ##网络卷积层改动
##2023年12月4日10点45