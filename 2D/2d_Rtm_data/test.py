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
from Model_2DUnet1118 import net
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
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
device="cuda"
model=net(2,1,128).to(device)
model=nn.parallel.DataParallel(model)
model.load_state_dict(torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_3.pkl"))


m=8
##data_prepare
k=29998
save=False
j=50
ny=nx=100
R=torch.from_file('/home/pengyaoguang/data/2D_data/2D_RTM/RTM{}_{}.bin'.format(k,j),
                    size=ny*nx).reshape(ny, nx)
label=torch.from_file('/home/pengyaoguang/data/2D_data/2D_v_model/v{}_{}.bin'.format(k,j),
        size=ny*nx).reshape(ny, nx)
label_smooth=torch.tensor(1/gaussian_filter(1/label, 40))
vmax=torch.max(R)
plt.figure()
plt.imshow(R.T/vmax,cmap="gray")
plt.colorbar()
if save:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/{}_{}rtm.png".format(k,j))
else:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/rtm0.png")


R1=R.reshape(1,1,R.shape[0],R.shape[1])
plt.figure()
plt.imshow(label.T)
plt.colorbar()
if save:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/{}_{}v_real_test.png".format(k,j))
else:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/v_real_test0.png")

label1=label.reshape(1,1,label.shape[0],label.shape[1])

plt.figure()
plt.imshow(label_smooth.T)
plt.colorbar()
if save:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/{}_{}v_smooth_test.png".format(k,j))
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
loss=loss_1(y_1,label1)
print(loss)
l1=loss_2(y_1,label1)
g=torch.zeros_like(y_1)
l2=loss_2(g,label1)
print("relative_error:",l1/l2)
plt.figure()
plt.imshow(y_1.detach().cpu()[0,0].T)
plt.colorbar()
if save:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/{}_{}v_updete_test.png".format(k,j))
else:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/v_updete_test0.png")
print('ssim',ssim_metric(label.detach().cpu().numpy(),y_1.detach().cpu()[0,0].numpy()))

plt.figure()
plt.imshow(y_1.detach().cpu()[0,0].T-label.detach().cpu().T)
plt.colorbar()
if save:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/{}_{}v_error.png".format(k,j))
else:
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/v_error0.png")
# sio.savemat("/home/pengyaoguang/data/3D_net_result/3D_result{}.mat".format(m),{'RTM':R,'v_real':label,'v_update':y_1.detach().cpu()[0,0],'v_smooth':label_smooth})