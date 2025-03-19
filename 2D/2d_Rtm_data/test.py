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
from Model_2DUnet1208 import net
import os 
from skimage.metrics import structural_similarity as ssim

def ssim_metric(target: object, prediction: object, win_size: int=21):
    cur_ssim = ssim(
        target,
        prediction,
        win_size=win_size,
        data_range=target.max() - target.min(),
    )
    return cur_ssim
import math

def SNR_singlech(S, SN):
    PS = torch.mean(torch.square(S))
    PN = torch.mean(torch.square(SN-S))
    # PS = torch.sum((S - mean_S) ** 2) # 纯信号的功率
    # PN = torch.sum((S - SN) ** 2) # 噪声的功率
    snr = 10 * math.log10(PS / PN) # 计算信噪比
    return snr

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device="cuda"

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
# model=torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_62.pkl")
# model=torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_49.pkl")
model=torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_59.pkl")
# model=net(2,1,128).to(device)
# model=nn.parallel.DataParallel(model)
# model.load_state_dict(torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_18.pkl")) 
m=8
##data_prepare
# k=25242
# k=1
k=201
j=80
# j=50
save=True  

ny=nx=100
##new_data
R=torch.Tensor(sio.loadmat("/home/pengyaoguang/data/3D_RTM2/RTM{}".format(k))["RTM"][20:120,20:120,20:120][j])
label=torch.Tensor(sio.loadmat("/home/pengyaoguang/data/3D_RTM2/v{}".format(k-1))["v"][j]*1000)
# R=torch.Tensor(sio.loadmat("/home/pengyaoguang/data/3D_RTM2/RTM{}".format(k))["RTM"][20:120,20:120,20:120][:,j])
# label=torch.Tensor(sio.loadmat("/home/pengyaoguang/data/3D_RTM2/v{}".format(k-1))["v"][:,j]*1000)
##process_real_data
# R=torch.Tensor(sio.loadmat("/home/pengyaoguang/data/real_data/RTM{}".format(k))["RTM"])
# label=torch.Tensor(sio.loadmat("/home/pengyaoguang/data/real_data/v{}".format(k))["v"]*1000)
# R=torch.Tensor(sio.loadmat("/home/pengyaoguang/data/3D_RTM/RTM{}".format(k))["RTM"][20:120,20:120,20:120][j])
# label=torch.Tensor(sio.loadmat("/home/pengyaoguang/data/3D_v_model/v{}".format(k))["v"][j]*1000)

# R=torch.from_file('/home/pengyaoguang/data/2D_data/2D_RTM1209/RTM{}.bin'.format(k),
#                 size=ny*nx).reshape(ny, nx)
# label=torch.Tensor(np.fromfile('/home/pengyaoguang/data/2D_data/2D_v_model1209/v{}.bin'.format(k))).reshape(ny, nx)*1000

# R=torch.from_file('/home/pengyaoguang/data/2D_data/2D_RTM/RTM{}_{}.bin'.format(k,j),
#                     size=ny*nx).reshape(ny, nx)
# label=torch.from_file('/home/pengyaoguang/data/2D_data/2D_v_model/v{}_{}.bin'.format(k,j),size=ny*nx).reshape(ny, nx)


label_smooth=torch.tensor(1/gaussian_filter(1/label, 40))
# label_smooth=torch.tensor(label)
vmax=torch.max(R)
plt.figure()
plt.imshow(R.T/vmax,cmap="gray")
plt.colorbar()
if save:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/{}_{}rtm.eps".format(k,j),dpi=300)
else:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/rtm0.png")
R1=R.reshape(1,1,R.shape[0],R.shape[1])
plt.figure()
plt.imshow(label.T,cmap='jet')
plt.colorbar()
if save:
    sio.savemat('/home/pengyaoguang/data/well_data/{}_{}v_real_test.mat'.format(k,j),{'data':label})
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/{}_{}v_real_test.eps".format(k,j),dpi=300)
else:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/v_real_test0.png")

label1=label.reshape(1,1,label.shape[0],label.shape[1])

plt.figure()
plt.imshow(label_smooth.T,cmap='jet')
plt.colorbar()
if save:
    sio.savemat('/home/pengyaoguang/data/well_data/{}_{}v_smooth_test.mat'.format(k,j),{'data':label_smooth})
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/{}_{}v_smooth_test.eps".format(k,j),dpi=300)
else:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/v_smooth_test0.png")

label_smooth1=label_smooth.reshape(1,1,label_smooth.shape[0],label_smooth.shape[1])
x=np.zeros((1,2,100,100))
x[:,0]=R1
x[:,1]=label_smooth1
# x,y=DataLoad(k,k)



##test
x=torch.from_numpy(x).float().to(device)
label1=label1.float().to(device)

# device="cuda"
# x,y=DataLoad(2)
# x=torch.from_numpy(x).float().to(device)
# y=torch.from_numpy(y).float().to(device)
# label1=y

loss_1=torch.nn.L1Loss()
loss_2=torch.nn.MSELoss()
y_1=model(x)
# print(total_variation_loss(y_1))
loss=loss_1(y_1,label1)
print(loss)
l1=loss_2(y_1,label1)
g=torch.zeros_like(y_1)
l2=loss_2(g,label1)
print("relative_error:",l1/l2)
snr=SNR_singlech(label1[0,0],y_1[0,0])
print('snr:',snr)
plt.figure()
plt.imshow(y_1.detach().cpu()[0,0].T,cmap='jet')
plt.colorbar()
if save:
    sio.savemat('/home/pengyaoguang/data/well_data/{}_{}v_updete_test.mat'.format(k,j),{'data':y_1.detach().cpu()[0,0]})
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/{}_{}v_updete_test.eps".format(k,j),dpi=300)
else:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/v_updete_test0.png")
print('ssim',ssim_metric(label.detach().cpu().numpy(),y_1.detach().cpu()[0,0].numpy()))

plt.figure()
plt.imshow(y_1.detach().cpu()[0,0].T-label.detach().cpu().T,cmap='jet')
plt.colorbar()
if save:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/{}_{}v_error.eps".format(k,j),dpi=300)
else:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/v_error0.png")
# sio.savemat("/home/pengyaoguang/data/3D_net_result/3D_result{}.mat".format(m),{'RTM':R,'v_real':label,'v_update':y_1.detach().cpu()[0,0],'v_smooth':label_smooth})

