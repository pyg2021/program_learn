#通过平层去预测断层以及褶皱层
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
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
start=time.time()

##data_prepare
BatchSize=8
device="cuda"
# x_1,y_1=DataLoad(0,0+99)
# x_2,y_2=DataLoad(5000,5000+99)
# x_3,y_3=DataLoad(10000,10000+99)
# x=np.concatenate((x_1,x_2,x_3),axis=0)
# y=np.concatenate((y_1,y_2,y_3),axis=0)
x,y=DataLoad(0,0+99)
trian_number=y.shape[0]
train_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
train_loader = data_utils.DataLoader(train_data,batch_size=BatchSize,shuffle=True)
# x=torch.from_numpy(x).float().to(device)
# y=torch.from_numpy(y).float().to(device)

x_1,y_1=DataLoad(100,110)
x_2,y_2=DataLoad(5000+100,5000+110)
x_3,y_3=DataLoad(10000+100,10000+110)
x=np.concatenate((x_1,x_2,x_3),axis=0)
y=np.concatenate((y_1,y_2,y_3),axis=0)
test_number=y.shape[0]
test_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
test_loader = data_utils.DataLoader(test_data,batch_size=BatchSize,shuffle=True)

model=net(2,1,True,True).to(device)
model=nn.parallel.DataParallel(model)

epoch=10000
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.8)
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
    sum_1=0
    for i,(x,y) in enumerate(train_loader):
        sum_1+=1
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        y_1=model(x)
        loss=loss_1(y_1,y)+2*loss_1(torch.clamp(y_1,1.5,8),y_1)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss+=loss.detach().cpu().item()
    epoch_loss=epoch_loss/sum_1
    loss_all.append(epoch_loss)
    

    test_loss=0
    sum_2=0
    with torch.no_grad():
        model.eval()
        for j,(x,y) in enumerate(test_loader):
            sum_2+=1
            x=x.to(device)
            y=y.to(device)
            y_1=model(x)
            loss=loss_1(y_1,y)+2*loss_1(torch.clamp(y_1,1.5,8),y_1)
            test_loss+=loss.detach().cpu().item()
    test_loss=test_loss/sum_2
    test_loss_all.append(test_loss)
    print(' epoch: ',epoch_i," train_loss: ",epoch_loss," test_loss: ",test_loss)
    if epoch_i%50==0 and epoch_i>=40:
        print((time.time()-start)/60,"s")
        plt.figure()
        plt.imshow(model(x).cpu().detach()[0,0,50,:,:].T)
        plt.colorbar()
        plt.savefig("/home/pengyaoguang/data/3D_net_result/v_updata6_5.png")
        plt.close()
        sio.savemat("/home/pengyaoguang/data/3D_net_result/v_updata6.mat",{"v":model(x).cpu().detach()[0,0]})
        torch.save(model.state_dict(),"/home/pengyaoguang/data/3D_net_model/modeltest6_5.pkl")

        plt.figure()
        plt.imshow(y.cpu().detach()[0,0,50,:,:].T)
        plt.colorbar()
        plt.savefig("/home/pengyaoguang/data/3D_net_result/v_real6_5.png")
        plt.close()

        plt.figure()
        plt.plot(range(len(loss_all)-40),loss_all[40:],label="train")
        plt.plot(range(len(test_loss_all)-40),test_loss_all[40:],label="test")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("/home/pengyaoguang/data/3D_net_result/history6_5.png")
        plt.close()

