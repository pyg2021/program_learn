import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
##data_prepare   def DataLoad(): 
def DataLoad(a,b):
    size=(b-a+1)*100
    x=np.zeros((size,2,100,100))
    y=np.zeros((size,1,100,100))
    i=0
    for k in range(a,b+1):
        R=sio.loadmat("/home/pengyaoguang/data/3D_RTM2/RTM{}".format(k))["RTM"][20:120,20:120,20:120].transpose(1,0,2)
        label=sio.loadmat("/home/pengyaoguang/data/3D_RTM2/v{}".format(k-1))["v"].transpose(1,0,2)*1000
        label_smooth=1/gaussian_filter(1/label,sigma=40)
        for j in range(100):
            R1=R[j].reshape(1,1,R.shape[1],R.shape[2])
            label1=label[j].reshape(1,1,label.shape[1],label.shape[2])
            label_smooth1=label_smooth[j].reshape(1,1,label_smooth.shape[1],label_smooth.shape[2])
            x[i,0]=R1
            x[i,1]=label_smooth1
            y[i,0]=label1
            i=i+1
    return x,y

