from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import scipy
import multiprocessing
from multiprocessing import Pool
# rtm=sio.loadmat("/home/pengyaoguang/data/3D_RTM2/RTM_overtrust2.mat")['RTM']
# rtm2=sio.loadmat("/home/pengyaoguang/data/3D_RTM2/RTM_overtrust{}.mat")['RTM']
# plt.imshow(rtm[50].T/np.max(rtm[50]))
# plt.colorbar()
# plt.savefig('/home/pengyaoguang/program_learn/program/shengli/RTM_data_prepare/2.png')
# plt.close()
# plt.imshow(rtm2[50].T/np.max(rtm2[50]))
# plt.colorbar()
# plt.savefig('/home/pengyaoguang/program_learn/program/shengli/RTM_data_prepare/3.png')
# plt.close()
v=sio.loadmat("/home/pengyaoguang/data/3D_v_model/fianl_v3.mat")['v']*1000
# sio.savemat("/home/pengyaoguang/data/3D_v_model/000.mat",{'v':v})
plt.imshow(v[50].T[::2,::2])
plt.colorbar()
plt.savefig('/home/pengyaoguang/program_learn/program/shengli/RTM_data_prepare/4.png')
plt.close()
# v3=sio.loadmat("/home/pengyaoguang/data/3D_v_model/fianl_v3.mat")["v"]
# v2=sio.loadmat("/home/pengyaoguang/data/3D_v_model/fianl_v2.mat")["v"]
# print(v2[:,:,99])
# number=0
# for i in range(v3.shape[0]):
#     for j in  range(v3.shape[1]):
#         for k in range(v3.shape[2]):
#             if v3[i][j][k]!=v2[i][j][k]:
#                 number+=1
#                 print(i,j,k,v3[i][j][k],v2[i][j][k])
# print(number)
# plt.imshow(v2[0].T)
# plt.colorbar()
# plt.savefig('/home/pengyaoguang/program_learn/program/shengli/RTM_data_prepare/2.png')
# plt.close()
# plt.imshow(v3[0].T)
# plt.colorbar()
# plt.savefig('/home/pengyaoguang/program_learn/program/shengli/RTM_data_prepare/3.png')