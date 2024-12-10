import torch
import scipy.io as sio
import numpy as np
from scipy.ndimage import gaussian_filter
##data_prepare   def DataLoad(): 
def DataLoad(a,b):
    size=(b-a+1)
    x=np.zeros((size,2,100,100))
    y=np.zeros((size,1,100,100))
    i=0
    ny=100
    nx=100
    for k in range(a,b+1):
        R=torch.from_file('/home/pengyaoguang/data/2D_data/2D_RTM1209/RTM{}.bin'.format(k),
                size=ny*nx).reshape(ny, nx)
        R1=R.reshape(1,1,R.shape[0],R.shape[1])
        label=np.fromfile('/home/pengyaoguang/data/2D_data/2D_v_model1209/v{}.bin'.format(k)).reshape(ny, nx)*1000
        label1=label.reshape(1,1,label.shape[0],label.shape[1])
        label_smooth=torch.tensor(1/gaussian_filter(1/label, 40))
        label_smooth1=label_smooth.reshape(1,1,label_smooth.shape[0],label_smooth.shape[1])
        x[i,0]=R1
        x[i,1]=label_smooth1
        y[i,0]=label1
        i=i+1
    return x,y
# x,y=DataLoad(0,1)
# print(x.shape,y.shape)