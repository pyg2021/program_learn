import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from examples.seismic import Model, plot_velocity
from devito import configuration
configuration['log-level'] = 'WARNING'
# Configure model presets
from examples.seismic import demo_model
import time
import scipy.io as sio

v=sio.loadmat("/home/pengyaoguang/data/fianl_v3.mat")["v"]
# m=58
start=0

#59-
# L=600
# for i in range(0,v.shape[0]-L,int(L/2)):
#     for j in range(0,v.shape[1]-L,int(L/2)):
#         m+=1
#         v_0=v[i:i+L,j:j+L][::4,::4]
#         # print(v_0.shape)
#         sio.savemat("/home/pengyaoguang/data/3D_RTM2/v{}.mat".format(m),{'v':v_0})
#1-49
# L=200
# for i in range(0,v.shape[0]-L,int(L/2)):
#     for j in range(0,v.shape[1]-L,int(L/2)):
#         m+=1
#         v_0=v[i:i+L,j:j+L][::2,::2]
#         # v_0.tofile("/home/pengyaoguang/data/3D_RTM2/v{}.bin".format(m))
#         sio.savemat("/home/pengyaoguang/data/3D_RTM2/v{}.mat".format(m),{'v':v_0})
#         print(v_0.shape)
# print(m)
L=400
m=58
#50-58

for i in range(0,v.shape[0]-L,int(L/6)):
    for j in range(0,v.shape[1]-L,int(L/6)):
        m+=1
        v_0=v[i:i+L,j:j+L][::4,::4]
        print(v_0.shape)
        # sio.savemat("/home/pengyaoguang/data/3D_RTM2/v{}.mat".format(m),{'v':v_0})
# print(m)
L=600
m=107
#50-58

for i in range(0,v.shape[0]-L,int(L/6)):
    for j in range(0,v.shape[1]-L,int(L/6)):
        m+=1
        v_0=v[i:i+L,j:j+L][::6,::6]
        print(v_0.shape)
        sio.savemat("/home/pengyaoguang/data/3D_RTM2/v{}.mat".format(m),{'v':v_0})
print(m)