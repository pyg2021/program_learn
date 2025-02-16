import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
# data=np.fromfile('/home/data/imp.npy')
data=np.load('/home/data/imp.npy')
# print(data.shape)
data=data.transpose(2,1,0)
# print(data.shape)
v=data*(4.8-1.5)+1.5
m=200
##502*501*400
v_0=v[::5,::5,::4][:100,:100]
sio.savemat("/home/pengyaoguang/data/3D_RTM2/v{}.mat".format(m),{'v':v_0})

m=201
L=200
#201-216
for i in range(0,v.shape[0]-L,int(L/2)):
    for j in range(0,v.shape[1]-L,int(L/2)):
        v_0=v[i:i+L,j:j+L][::2,::2,::4]
        print(v_0.shape,m)
        # plt.imshow(v_0[0].T)
        # plt.colorbar()
        # plt.savefig('/home/pengyaoguang/program_learn/2D/0.png')
        # plt.close()
        # time.sleep(1)
        # break
        # sio.savemat("/home/pengyaoguang/data/3D_RTM2/v{}.mat".format(m),{'v':v_0})
        m+=1
# print(v.shape)
# plt.imshow(v.T)
# plt.colorbar()
# plt.savefig('/home/pengyaoguang/program_learn/2D/0.png')
# plt.close()
# print(v.shape)
# sio.savemat("/home/pengyaoguang/data/3D_RTM2/v{}.mat".format(m),{'v':v})
    # break
