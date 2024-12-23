#新的合成数据训练instance+128
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
from DataLoad1210 import DataLoad as DataLoad1
from DataLoad1221 import DataLoad as DataLoad2
from Model_2DUnet1208 import net
import os 
from skimage.metrics import structural_similarity as ssim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
start=time.time()

##data_prepare
BatchSize=300

device="cuda"
# x_1,y_1=DataLoad1(0,8000)
# x_2,y_2=DataLoad(25000,25299)
# x_3,y_3=DataLoad(25000,25000)
# x=np.concatenate((x_1,x_2,x_3),axis=0)
# y=np.concatenate((y_1,y_2,y_3),axis=0)
x,y=DataLoad2(25000,25250)
trian_number=y.shape[0]
train_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
train_loader_1 = data_utils.DataLoader(train_data,batch_size=BatchSize,shuffle=True)

# x_1,y_1=DataLoad1(8000,9000)
# x_2,y_2=DataLoad(25000,25000)
# x_3,y_3=DataLoad(25000,25000)
# x=np.concatenate((x_1,x_2,x_3),axis=0)
# y=np.concatenate((y_1,y_2,y_3),axis=0)
x,y=DataLoad2(25000+250,25000+279)
test_number=y.shape[0]
test_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
test_loader_1 = data_utils.DataLoader(test_data,batch_size=BatchSize,shuffle=True)


# x_1,y_1=DataLoad(25000+0,25000+1)
# x,y=x_1,y_1
# trian_number=y.shape[0]
# train_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
# train_loader_2 = data_utils.DataLoader(train_data,batch_size=BatchSize,shuffle=True)

# x_1,y_1=DataLoad(25000+0,25000+1)
# x,y=x_1,y_1
# test_number=y.shape[0]
# test_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
# test_loader_2 = data_utils.DataLoader(test_data,batch_size=BatchSize,shuffle=True)




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
    number=0
    loss_number=float('inf')
    lr=1e-3
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
            loss=loss_1(y_1,y)+loss_1(torch.clamp(y_1,1000,10000),y_1)
            # loss=loss_1(y_1,y)
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
                loss=loss_1(y_1,y)
                test_loss+=loss.detach().cpu().item()
            a=0.1
            if test_loss+a>=loss_number:
                number+=1
                if number>100:
                    number=0
                    lr=lr*0.5
                    print('----------------------------------------------')
                    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
                    # loss_number=float('inf')
                    # model.load_state_dict(torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_{}.pkl".format(save_number)))
            else:
                number=0
                loss_number=test_loss
                torch.save(model.state_dict(),"/home/pengyaoguang/data/2D_data/2D_result/modeltest9_{}.pkl".format(save_number))
        test_loss=test_loss/sum_2
        test_loss_all.append(test_loss)
        print(' epoch: ',epoch_i," train_loss: ",epoch_loss," test_loss: ",test_loss)
        # test(model,train_loader,loss_1,device)
        # test(model,test_loader,loss_1,device)
        if epoch_i%2==0 and epoch_i>20:
            print((time.time()-start)/60,"min")
            plt.figure()
            plt.imshow(model(x).cpu().detach()[0,0,:,:].T)
            plt.colorbar()
            plt.savefig("/home/pengyaoguang/data/2D_data/2D_result/v_updata9_{}.png".format(save_number))
            plt.close()
            plt.figure()
            plt.imshow(x.cpu().detach()[0,1,:,:].T)
            plt.colorbar()
            plt.savefig("/home/pengyaoguang/data/2D_data/2D_result/v_start9_{}.png".format(save_number))
            plt.close()
            # sio.savemat("/home/pengyaoguang/data/2D_data/2D_result/v_updata9_{}.mat".format(save_number),{"v":model(x).cpu().detach()[0,0]})
            # if test_loss==
            # torch.save(model.state_dict(),"/home/pengyaoguang/data/2D_data/2D_result/modeltest9_{}.pkl".format(save_number))

            plt.figure()
            plt.imshow(y.cpu().detach()[0,0,:,:].T)
            plt.colorbar()
            plt.savefig("/home/pengyaoguang/data/2D_data/2D_result/v_real9_{}.png".format(save_number))
            plt.close()

            plt.figure()
            plt.plot(range(len(loss_all)-100),loss_all[100:],label="train")
            plt.plot(range(len(test_loss_all)-100),test_loss_all[100:],label="test")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.savefig("/home/pengyaoguang/data/2D_data/2D_result/history9_{}.png".format(save_number))
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
            loss=loss_1(y_1,y)+2*loss_1(torch.clamp(y_1,1000,8000),y_1)
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

model=net(2,1,128).to(device)
model=nn.parallel.DataParallel(model)
# ewc=EWC(model, train_loader_1, device)
model.load_state_dict(torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_18.pkl"))


optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.7)
# loss_1=torch.nn.L1Loss()
loss_1=torch.nn.MSELoss()
train(model,train_loader_1,test_loader_1,10000,device,optimizer,scheduler,loss_1,save_number=18)
# test(model,train_loader_1,loss_1,device)
# test(model,train_loader_2,loss_1,device)


