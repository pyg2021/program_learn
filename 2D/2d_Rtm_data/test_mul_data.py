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
from Model_2DUnet1118 import diffusion_net
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


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device="cuda"
model=diffusion_net(2,1).to(device)
model=nn.parallel.DataParallel(model)
model.load_state_dict(torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_4.pkl"))
# x_1,y_1=DataLoad(25000+200,25000+240)
# x_2,y_2=DataLoad(30000+80,30000+100)
x_1,y_1=DataLoad(25000+0,25000+200)
x_2,y_2=DataLoad(30000+0,30000+200)
x_3,y_3=DataLoad(25000+0,25000+1)
x=np.concatenate((x_1,x_2,x_3),axis=0)
y=np.concatenate((y_1,y_2,y_3),axis=0)
# x,y=DataLoad(25000+80,25000+100)
test_number=y.shape[0]
test_data=data_utils.TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).float())
test_loader_1 = data_utils.DataLoader(test_data,batch_size=100,shuffle=True)
model.eval()
loss_1=torch.nn.MSELoss()
test_loss=0
for j,(x,y) in enumerate(test_loader_1):
    x=x.to(device)
    y=y.to(device)
    y_1=model(x)
    loss=loss_1(y_1,y)+2*loss_1(torch.clamp(y_1,1000,10000),y_1)
    test_loss+=loss.detach().cpu().item()
print(test_loss/(j+1))