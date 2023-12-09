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
from Model3D_unt import net
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = "0,2"
start=time.time()

##data_prepare
BatchSize=8
device="cuda"
x_1,y_1=DataLoad(0,0+10)
x_2,y_2=DataLoad(5000,5000+0)
x_3,y_3=DataLoad(10000,10000+0)
# x=np.zeros((x_1.shape[0]+x_2.shape[0]+x_3.shape[0],2,100,100,100))
# y=np.zeros((y_1.shape[0]+y_2.shape[0]+y_3.shape[0],1,100,100,100))
x=np.concatenate((x_1,x_2,x_3),axis=0)
y=np.concatenate((y_1,y_2,y_3),axis=0)


# x,y=DataLoad(10000,10000+0)


train_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
train_loader = data_utils.DataLoader(train_data,batch_size=BatchSize,shuffle=True)
# x=torch.from_numpy(x).float().to(device)
# y=torch.from_numpy(y).float().to(device)

x_1,y_1=DataLoad(100,100+0)
x_2,y_2=DataLoad(5009,5009+0)
x_3,y_3=DataLoad(10009,10009+0)
x=np.concatenate((x_1,x_2,x_3),axis=0)
y=np.concatenate((y_1,y_2,y_3),axis=0)
test_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
test_loader = data_utils.DataLoader(test_data,batch_size=BatchSize,shuffle=True)

model=net(2,1,True,True).to(device)
model=nn.parallel.DataParallel(model)
# model.load_state_dict(torch.load("/home/pengyaoguang/data/3D_net_model/modeltest6_3.pkl"))

epoch=10000
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=800,gamma=0.8)
loss_1=torch.nn.L1Loss()
# plt.figure()
# plt.imshow(y.cpu().detach()[0,0,50,:,:].T)
# plt.colorbar()
# plt.savefig("/home/pengyaoguang/data/3D_net_result/v_real.png")
# plt.close()

loss_all=[]
test_loss_all=[]
for epoch_i in range(epoch):
    epoch_loss=0
    model.train()
    for i,(x,y) in enumerate(train_loader):
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        y_1=model(x)
        loss=loss_1(y_1,y)+2*loss_1(torch.clamp(y_1,1.5,8),y_1)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss+=loss.detach().cpu().item()
    loss_all.append(epoch_loss)
    

    test_loss=0
    with torch.no_grad():
        model.eval()
        for j,(x,y) in enumerate(test_loader):
            x=x.to(device)
            y=y.to(device)
            y_1=model(x)
            loss=loss_1(y_1,y)+2*loss_1(torch.clamp(y_1,1.5,8),y_1)
            test_loss+=loss.detach().cpu().item()
    test_loss_all.append(test_loss)
    print(' epoch: ',epoch_i," train_loss: ",epoch_loss," test_loss: ",test_loss)
    if epoch_i%50==0 and epoch_i>=40:
        print(time.time()-start,"s")
        plt.figure()
        plt.imshow(model(x).cpu().detach()[0,0,50,:,:].T)
        plt.colorbar()
        plt.savefig("/home/pengyaoguang/data/3D_net_result/v_updata6_4.png")
        plt.close()
        sio.savemat("/home/pengyaoguang/data/3D_net_result/v_updata6.mat",{"v":model(x).cpu().detach()[0,0]})
        torch.save(model.state_dict(),"/home/pengyaoguang/data/3D_net_model/modeltest6_4.pkl")

        plt.figure()
        plt.imshow(y.cpu().detach()[0,0,50,:,:].T)
        plt.colorbar()
        plt.savefig("/home/pengyaoguang/data/3D_net_result/v_real6_4.png")
        plt.close()

        plt.figure()
        plt.plot(range(len(loss_all)-10),loss_all[10:],label="train")
        plt.plot(range(len(test_loss_all)-10),test_loss_all[10:],label="test")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("/home/pengyaoguang/data/3D_net_result/history6_4.png")
        plt.close()

