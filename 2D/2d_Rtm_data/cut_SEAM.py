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
# sio.savemat("/home/pengyaoguang/data/3D_RTM2/v{}.mat".format(m),{'v':v_0})
for i in range(v_0.shape[1]):
    if i%5==0:
        plt.figure()
        plt.imshow(v_0[:,i].T)
        plt.colorbar()
        plt.savefig('/home/pengyaoguang/program_learn/2D/0.png')
        plt.close()
        time.sleep(1)
m=201
L=200
#201-216
for i in range(0,v.shape[0]-L,int(L/2)):
    for j in range(0,v.shape[1]-L,int(L/2)):
        v_0=v[i:i+L,j:j+L][::2,::2,::4]
        # print(v_0.shape,m,i+L,j+L)
        # plt.imshow(v_0[0].T)
        # plt.colorbar()
        # plt.savefig('/home/pengyaoguang/program_learn/2D/0.png')
        # plt.close()
        # time.sleep(1)
        # break
        # sio.savemat("/home/pengyaoguang/data/3D_RTM2/v{}.mat".format(m),{'v':v_0})
        m+=1
m=217
L=300
##217-220
for i in range(0,v.shape[0]-L,int(L/2)):
    for j in range(0,v.shape[1]-L,int(L/2)):
        v_0=v[i:i+L,j:j+L][::3,::3,::4]
        # print(v_0.shape,m,i+L,j+L)
        # sio.savemat("/home/pengyaoguang/data/3D_RTM2/v{}.mat".format(m),{'v':v_0})
        m+=1
m=221
L=100
##221-245
for i in range(0,v.shape[0]-L,int(L)):
    for j in range(0,v.shape[1]-L,int(L)):
        v_0=v[i:i+L,j:j+L][::1,::1,::4]
        # print(v_0.shape,m,i+L,j+L)
        # sio.savemat("/home/pengyaoguang/data/3D_RTM2/v{}.mat".format(m),{'v':v_0})
        m+=1
m=246
L=400
##221-245
for i in range(v.shape[0]-1,L-1,-int(L)//6):
    for j in range(v.shape[1]-1,L-1,-int(L)//6):
        v_0=v[i-L+1:i+1,j+1-L:j+1][::4,::4,::4]
        print(v_0.shape,m,i,j)
        # sio.savemat("/home/pengyaoguang/data/3D_RTM2/v{}.mat".format(m),{'v':v_0})
        m+=1
