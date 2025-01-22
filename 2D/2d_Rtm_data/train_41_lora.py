#针对于三维的数据，通过将overtrust数据的井数据得到微调后的结果（并改变了对encodeing参数的学习率）,只设定5口井，均匀分布，数据量1000，三种数据块共存,通过少量真实数据微调
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
from DataLoad1231 import DataLoad as DataLoad3
from Model_2DUnet1208 import net
import os 
from skimage.metrics import structural_similarity as ssim
import random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
start=time.time()

##data_prepare
BatchSize=10

device="cuda"
# x_1,y_1=DataLoad3(2,27)
# x_2,y_2=DataLoad3(28,49)
# x_3,y_3=DataLoad3(60,117)
# x=np.concatenate((x_1,x_2,x_3),axis=0)
# y=np.concatenate((y_1,y_2,y_3),axis=0)
x,y=DataLoad3(1,1)
# x=x[::10]
# y=y[::10]
trian_number=y.shape[0]
train_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
train_loader_1 = data_utils.DataLoader(train_data,batch_size=BatchSize,shuffle=True)

x_1,y_1=DataLoad3(2,27)
x_2,y_2=DataLoad3(28,49)
x_3,y_3=DataLoad3(60,117)
x=np.concatenate((x_1,x_2,x_3),axis=0)
y=np.concatenate((y_1,y_2,y_3),axis=0)
# x,y=DataLoad3(1,1)
x=x[::100]
y=y[::100]
trian_number=y.shape[0]
train_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
train_loader_2 = data_utils.DataLoader(train_data,batch_size=BatchSize,shuffle=True)
# x_1,y_1=DataLoad(30009,30010)
# x_2,y_2=DataLoad(30000,30000)
# x_3,y_3=DataLoad(30000,30000)
# x=np.concatenate((x_1,x_2,x_3),axis=0)
# y=np.concatenate((y_1,y_2,y_3),axis=0)
x,y=DataLoad3(1,1)
# x=x[:2]
# y=y[:2]
# x,y=DataLoad(25000+80,25000+100)
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
    sample_list = [i for i in range(100)]
    state=-1
    for epoch_i in range(epoch):
        epoch_loss=0
        model.train()
        

        for i,(x,y) in enumerate(train_loader_2):
            x=x.to(device)
            y=y.to(device)
            optimizer.zero_grad()
            y_1=model(x)
            # loss=loss_1(y_1,y)+loss_1(torch.clamp(y_1,1000,10000),y_1)
            # tv_loss=total_variation_loss(y_1)
            # sam=random.sample(sample_list,5)
            # loss=loss_1(y_1[:,:,sam,:],y[:,:,sam,:])
            loss=loss_1(y_1,y)
            if ewc is not None:
                ewc_loss = ewc.penalty(model)
                loss += ewc_lambda * ewc_loss
                print('ewc:',ewc_lambda * ewc_loss)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            epoch_loss+=loss.detach().cpu().item()
        for i,(x,y) in enumerate(train_loader):
            x=x.to(device)
            y=y.to(device)
            optimizer.zero_grad()
            y_1=model(x)
            # loss=loss_1(y_1,y)+loss_1(torch.clamp(y_1,1000,10000),y_1)
            # tv_loss=total_variation_loss(y_1)
            # sam=random.sample(sample_list,5)
            # loss=loss_1(y_1[:,:,sam,:],y[:,:,sam,:])
            loss=loss_1(y_1[:,:,::5,:],y[:,:,::5,:])
            if ewc is not None:
                ewc_loss = ewc.penalty(model)
                loss += ewc_lambda * ewc_loss
                print('ewc:',ewc_lambda * ewc_loss)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            epoch_loss+=loss.detach().cpu().item()
        epoch_loss=epoch_loss/(i+1)
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
                if number>200:
                    number=0
                    print('----------------------------------------------')
                    # optimizer = torch.optim.AdamW(model.parameters(),lr=1e-2)
                    optimizer = torch.optim.AdamW([
                        {'params': [param for name, param in model.named_parameters() if ('A' not in name and 'B' not in name and 's' not in name) and ('down' in name or 'in_conv' in name)], 'lr': 0},
                        {'params': [param for name, param in model.named_parameters() if ('A' not in name and 'B' not in name and 's' not in name) and ('down' not in name and 'in_conv' not in name)], 'lr': 1e-3},  # 冻结原始权重
                        {'params': [param for name, param in model.named_parameters() if 'A' in name or 'B' in name or 's' in name], 'lr': 5e-1} # 优化LORA参数
                    ], lr=5e-1)
                    if state==1:
                        loss_1=torch.nn.MSELoss()
                    else:
                        loss_1=torch.nn.L1Loss()
                    state=-state
                    # loss_number=float('inf')
                    # model.load_state_dict(torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_{}.pkl".format(save_number)))
            else:
                number=0
                loss_number=test_loss
                torch.save(model,"/home/pengyaoguang/data/2D_data/2D_result/modeltest9_{}.pkl".format(save_number))
        test_loss=test_loss/sum_2
        test_loss_all.append(test_loss)
        print(' epoch: ',epoch_i," train_loss: ",epoch_loss," test_loss: ",test_loss)
        # test(model,train_loader,loss_1,device)
        # test(model,test_loader,loss_1,device)
        if epoch_i%2==0 and epoch_i>2:
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
            

            plt.figure()
            plt.imshow(y.cpu().detach()[0,0,:,:].T)
            plt.colorbar()
            plt.savefig("/home/pengyaoguang/data/2D_data/2D_result/v_real9_{}.png".format(save_number))
            plt.close()

            plt.figure()
            plt.plot(range(len(loss_all)-50),loss_all[50:],label="train")
            plt.plot(range(len(test_loss_all)-50),test_loss_all[50:],label="test")
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


# ewc=EWC(model, train_loader_1, device)

# 定义一个包装器（wrapper），用于在卷积层上应用LORA更新
class LORAConv2d(nn.Module):
    def __init__(self, original_conv, rank, downsample_factor=16):
        super(LORAConv2d, self).__init__()
        self.original_conv = original_conv
        self.rank = rank
        self.downsample_factor = downsample_factor
        
        # 初始化LORA参数
        self.A = nn.Parameter(torch.randn(original_conv.weight.size(0), rank) / downsample_factor)
        self.B = nn.Parameter(torch.randn(rank, original_conv.weight.size(1)) / downsample_factor)
        self.s = nn.Parameter(torch.zeros(1))  # 可选的缩放参数
 
    def forward(self, x):
        # 计算低秩更新项
        # delta_weight = torch.matmul(self.A, self.B) * self.s
        delta_weight = torch.matmul(self.A, self.B) * torch.sigmoid(self.s)
        # delta_weight = torch.matmul(self.A, self.B)
        # 注意：这里我们不能直接修改self.original_conv.weight，因为nn.Parameter是不可变的。
        # 相反，我们会在每次前向传播时创建一个新的卷积层，使用更新后的权重。
        
        # 创建一个新的卷积层，使用原始权重加上LORA更新项
        updated_weight = self.original_conv.weight + delta_weight
        updated_conv = nn.Conv2d(
            in_channels=self.original_conv.in_channels,
            out_channels=self.original_conv.out_channels,
            kernel_size=self.original_conv.kernel_size,
            stride=self.original_conv.stride,
            padding=self.original_conv.padding,
            dilation=self.original_conv.dilation,
            groups=self.original_conv.groups,
            bias=self.original_conv.bias is not None,
            padding_mode=self.original_conv.padding_mode
        )
        updated_conv.weight = nn.Parameter(updated_weight)
        if self.original_conv.bias is not None:
            updated_conv.bias = self.original_conv.bias
        
        # 使用更新后的卷积层进行前向传播
        return updated_conv(x)
 
# 包装U-Net中的卷积层以应用LORA
def apply_lora_to_unet(model, rank, downsample_factor=16):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            # 替换卷积层为LORAConv2d包装器
            setattr(model, name, LORAConv2d(module, rank, downsample_factor))
def total_variation_loss(image, weight=1.0):
    # 获取图像的形状
    batch_size, channels, height, width = image.size()
    
    # 计算水平方向的梯度
    horizontal_diff = image[:, :, :, 1:] - image[:, :, :, :-1]
    # 计算垂直方向的梯度
    vertical_diff = image[:, :, 1:, :] - image[:, :, :-1, :]
    
    # 计算梯度的绝对值
    abs_horizontal_diff = torch.abs(horizontal_diff)
    abs_vertical_diff = torch.abs(vertical_diff)
    
    # 计算TV正则化项（L1范数）
    tv_loss = weight * (torch.sum(abs_horizontal_diff) + torch.sum(abs_vertical_diff)) / (batch_size * channels * height * width - channels * (height + width - 1))
    
    # 注意：分母中的减项是为了去除边缘像素，因为这些像素在某一方向上没有相邻像素
    # 如果你希望包括边缘像素，可以调整分母的计算方式
    
    return tv_loss
# 应用LORA到预训练的U-Net模型
model=net(2,1,128).to(device)
model=nn.parallel.DataParallel(model)
model.load_state_dict(torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_18.pkl"))

# # model.eval()
apply_lora_to_unet(model, rank=16,downsample_factor=16)
# model=torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_22.pkl")
# model=nn.parallel.DataParallel(model)
# torch.save(model.state_dict(),"/home/pengyaoguang/data/2D_data/2D_result/modeltest9_{}.pkl".format(10))
# model.load_state_dict(torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_10.pkl"))
# torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_10.pkl").keys()==model.state_dict().keys()
# 现在，pretrained_model中的卷积层已经被LORAConv2d包装器替换，
# 在前向传播时会动态地应用LORA更新。
# model.load_state_dict(torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_10.pkl"))
# 定义损失函数和优化器（注意：这里我们不会优化原始权重，只会优化LORA参数）
# optimizer = torch.optim.AdamW([
#     {'params': [param for name, param in model.named_parameters() if 'A' not in name and 'B' not in name and 's' not in name], 'lr': 1e-2},  # 冻结原始权重
#     {'params': [param for name, param in model.named_parameters() if 'A' in name or 'B' in name or 's' in name], 'lr': 5e-1} # 优化LORA参数
# ], lr=5e-1)
optimizer = torch.optim.AdamW([
    {'params': [param for name, param in model.named_parameters() if ('A' not in name and 'B' not in name and 's' not in name) and ('down' in name or 'in_conv' in name or 'up' in name )], 'lr': 1e-3},
    {'params': [param for name, param in model.named_parameters() if ('A' not in name and 'B' not in name and 's' not in name) and ('down' not in name and 'in_conv' not in name and 'up' not in name)], 'lr': 1e-3},  # 冻结原始权重
    {'params': [param for name, param in model.named_parameters() if 'A' in name or 'B' in name or 's' in name], 'lr': 5e-1} # 优化LORA参数
], lr=5e-1)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.7)
# loss_1=torch.nn.L1Loss()
loss_1=torch.nn.MSELoss()
train(model,train_loader_1,test_loader_1,10000,device,optimizer,scheduler,loss_1,save_number=41)
# test(model,train_loader_1,loss_1,device)
# test(model,train_loader_2,loss_1,device)


