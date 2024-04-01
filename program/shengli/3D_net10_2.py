import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from DataLoad_well_2 import DataLoad
from Model3D_unt3 import net
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
start=time.time()

##data_prepare
BatchSize=8
device="cuda"
x_1,y_1=DataLoad(25000+0,25000+99)
x_2,y_2=DataLoad(25000+100,25000+199)
x_3,y_3=DataLoad(25000+199,25000+199)
x=np.concatenate((x_1,x_2,x_3),axis=0)
y=np.concatenate((y_1,y_2,y_3),axis=0)
# x,y=DataLoad(30000+0,30000+80)
trian_number=y.shape[0]
train_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
train_loader_1 = data_utils.DataLoader(train_data,batch_size=BatchSize,shuffle=True)

# x_1,y_1=DataLoad(15000+91,15000+100)
# x_2,y_2=DataLoad(20000+100,20000+108)
# x_3,y_3=DataLoad(5000+100,5000+109)
# x=np.concatenate((x_1,x_2,x_3),axis=0)
# y=np.concatenate((y_1,y_2,y_3),axis=0)
x,y=DataLoad(25000+220,25000+240)
test_number=y.shape[0]
test_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
test_loader_1 = data_utils.DataLoader(test_data,batch_size=BatchSize,shuffle=True)


x_1,y_1=DataLoad(15000,15000+1)
x,y=x_1,y_1
trian_number=y.shape[0]
train_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
train_loader_2 = data_utils.DataLoader(train_data,batch_size=BatchSize,shuffle=True)

x_1,y_1=DataLoad(15000,15000+1)
x,y=x_1,y_1
test_number=y.shape[0]
test_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
test_loader_2 = data_utils.DataLoader(test_data,batch_size=BatchSize,shuffle=True)

model=net(3,1,True,True).to(device)
model=nn.parallel.DataParallel(model)


# plt.figure()
# plt.imshow(y.cpu().detach()[0,0,50,:,:].T)
# plt.colorbar()
# plt.savefig("/home/pengyaoguang/data/3D_net_result/v_real.png")
# plt.close()

# EWC implementation
class EWC:
    def __init__(self, model, dataloader, device, importance=1000):
        self.model = model
        self.importance = importance
        self.device = device
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)
#计算fisher信息矩阵
    def _compute_fisher(self, dataloader):
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p.data)

        self.model.train()
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            # output = F.log_softmax(self.model(data), dim=1)
            # target=  F.log_softmax(target, dim=1)
            output=self.model(data)
            # output=output.reshape(output.shape[0],output.shape[2]*output.shape[3]*output.shape[4])
            # target=target.reshape(target.shape[0],target.shape[2]*target.shape[3]*target.shape[4])
            # loss = F.nll_loss(output, target)
            loss_1=torch.nn.L1Loss()
            loss = loss_1(output, target)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    fisher[n] += (p.grad ** 2) / len(dataloader)

        return fisher

    def penalty(self, new_model):
        loss = 0
        for n, p in new_model.named_parameters():
            if p.requires_grad:
                _loss = self.fisher[n] * (p - self.params[n]) ** 2
                loss += _loss.sum()
        return loss * (self.importance / 2)

def train(model,train_loader,test_loader,epoch,device,optimizer,scheduler,loss_1,ewc=None, ewc_lambda=0.5,save_number=0):
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
            if ewc is not None:
                ewc_loss = ewc.penalty(model)
                loss += ewc_lambda * ewc_loss
                print('ewc:',ewc_lambda * ewc_loss)
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
        # test(model,train_loader_1,loss_1,device)
        # test(model,test_loader_2,loss_1,device)
        if epoch_i%5==0 and epoch_i>=0:
            print((time.time()-start)/60,"min")
            plt.figure()
            plt.imshow(model(x).cpu().detach()[0,0,50,:,:].T)
            plt.colorbar()
            plt.savefig("/home/pengyaoguang/data/3D_net_result/v_updata10_{}.png".format(save_number))
            plt.close()
            sio.savemat("/home/pengyaoguang/data/3D_net_result/v_updata10_{}.mat".format(save_number),{"v":model(x).cpu().detach()[0,0]})
            torch.save(model.state_dict(),"/home/pengyaoguang/data/3D_net_model/modeltest10_{}.pkl".format(save_number))

            plt.figure()
            plt.imshow(y.cpu().detach()[0,0,50,:,:].T)
            plt.colorbar()
            plt.savefig("/home/pengyaoguang/data/3D_net_result/v_real10_{}.png".format(save_number))
            plt.close()

            plt.figure()
            plt.plot(range(len(loss_all)),loss_all[:],label="train")
            plt.plot(range(len(test_loss_all)),test_loss_all[:],label="test")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.savefig("/home/pengyaoguang/data/3D_net_result/history10_{}.png".format(save_number))
            plt.close()
def test(model,test_loader,loss_1,device,save_number=0):
    test_loss_all=[]
    test_loss=0
    sum_2=0
    with torch.no_grad():
        model.eval()
        for j,(x,y) in enumerate(test_loader):
            sum_2+=1
            x=x.to(device)
            y=y.to(device)
            y_1=model(x)
            loss=loss_1(y_1,y)+loss_1(torch.clamp(y_1,1.5,8),y_1)
            test_loss+=loss.detach().cpu().item()
    test_loss=test_loss/sum_2
    test_loss_all.append(test_loss)
    print(" test_loss: ",test_loss)


    # plt.figure()
    # plt.imshow(model(x).cpu().detach()[0,0,50,:,:].T)
    # plt.colorbar()
    # plt.savefig("/home/pengyaoguang/data/3D_net_result/v_updata8_{}.png".format(save_number))
    # plt.close()
    # sio.savemat("/home/pengyaoguang/data/3D_net_result/v_updata8_{}.mat".format(save_number),{"v":model(x).cpu().detach()[0,0]})

    # plt.figure()
    # plt.imshow(y.cpu().detach()[0,0,50,:,:].T)
    # plt.colorbar()
    # plt.savefig("/home/pengyaoguang/data/3D_net_result/v_real8_{}.png".format(save_number))
    # plt.close()


# model.load_state_dict(torch.load("/home/pengyaoguang/data/3D_net_model/modeltest6_5.pkl"))
# optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
# scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=300,gamma=0.6)
# loss_1=torch.nn.L1Loss()
# # train(model,train_loader_1,test_loader,1000,device,optimizer,scheduler,loss_1,save_number=13)
# test(model,train_loader_1,loss_1,device)
# test(model,train_loader_2,loss_1,device)


# ewc=EWC(model, train_loader_1, device)
# model.load_state_dict(torch.load("/home/pengyaoguang/data/3D_net_model/modeltest9_10.pkl"))
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-2)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.5)
# loss_1=torch.nn.L1Loss()
loss_1=torch.nn.L1Loss()
train(model,train_loader_1,test_loader_1,4000,device,optimizer,scheduler,loss_1,save_number=2)
# test(model,train_loader_1,loss_1,device)
# test(model,train_loader_2,loss_1,device)