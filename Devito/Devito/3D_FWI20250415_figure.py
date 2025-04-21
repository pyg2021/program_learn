import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
from examples.seismic import Model, plot_velocity
start= time.time()
from devito import configuration
import torch
configuration['log-level'] = 'WARNING'
m=25242
# m=1
# m=201
n=50
label_upadte=sio.loadmat("/home/yaoguang/data/3D_FWI/v_update{}_{}.mat".format(m,n))["v"][tuple(slice(20, -20) for _ in range(3))][n]
label=sio.loadmat("/home/yaoguang/data/3D_v_model/v{}.mat".format(m))['v'][n]
# label=sio.loadmat("/home/yaoguang/data/3D_RTM2/v{}.mat".format(m-1))['v'][n]
mi=np.min(label)/1000
ma=np.max(label)/1000
plt.figure()
plt.imshow(label.T/1000,cmap='jet',vmin=mi,vmax=ma)
clb=plt.colorbar(label='velocity(km/s)')
plt.xlabel('X(km)',fontsize=14)
plt.ylabel('Z(km)',fontsize=14)
plt.yticks(np.arange(0,100,step=19.9),['0','0.2','0.4','0.6','0.8','1.0'],fontsize=14)
plt.xticks(np.arange(0,100,step=19.9),['0','0.2','0.4','0.6','0.8','1.0'],fontsize=14)
plt.savefig('/home/yaoguang/data/3D_FWI/result/v_real{}_{}.png'.format(m,n))
plt.close()

plt.figure()
plt.imshow(label_upadte.T/1000,cmap='jet',vmin=mi,vmax=ma)
clb=plt.colorbar(label='velocity(km/s)')
plt.xlabel('X(km)',fontsize=14)
plt.ylabel('Z(km)',fontsize=14)
plt.yticks(np.arange(0,100,step=19.9),['0','0.2','0.4','0.6','0.8','1.0'],fontsize=14)
plt.xticks(np.arange(0,100,step=19.9),['0','0.2','0.4','0.6','0.8','1.0'],fontsize=14)
plt.savefig('/home/yaoguang/data/3D_FWI/result/v_upadte{}_{}.png'.format(m,n))
plt.close()