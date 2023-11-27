import torch
from Model import net
import torch.nn as nn
import scipy.io as sio
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
##data_prepare   def DataLoad(): 

def DataLoad(size):
    x=np.zeros((size,25,100,100,252))
    y=np.zeros((size,1,100,100,100))
    for k in range(size):
        for i in range(25):
            R=sio.loadmat("/home/pengyaoguang/data/3D_seismic_data/seismic{}_{}.mat".format(k,i))["seismic_data"]
            R1=R.reshape(1,1,R.shape[0],R.shape[1],R.shape[2])
            x[k,i]=R1
        label=sio.loadmat("/home/pengyaoguang/data/3D_v_model/v{}".format(k))["v"]
        label1=label.reshape(1,1,label.shape[0],label.shape[1],label.shape[2])
        label_smooth=1/gaussian_filter(1/label,sigma=10)
        label_smooth1=label_smooth.reshape(1,1,label_smooth.shape[0],label_smooth.shape[1],label_smooth.shape[2])
        
        # x[k,-1]=label_smooth1
        y[k,0]=label1
    return x,y