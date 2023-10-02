import numpy as np
import math
import torch
def cut_direct_wave_easy(data,i,source_locations,receiver_locations,v,dt):
    d_rec=receiver_locations[:,:,1]-receiver_locations[:,:,0]
    a=math.floor(d_rec/v/dt)
    b=3200
    t=data[i:,:]
    for k in range(t.shape[0]):
        t[k,:a*k+b]=0
    return t