import scipy.io as sio
import scipy
import numpy as np
import matplotlib.pyplot as plt
vel=sio.loadmat("/home/pengyaoguang/data/3D_v_model/Overthrust_vel.mat")["vel"]/1000

print(vel.shape)
m=0
for i in range(2):
    if i==0:
        v0=vel[100*i:100*i+100]
        for j in range(8):
            for k in range(8):
                v=v0[:,100*j:100*j+100,100*k:100*k+100]
                print(v.shape)
                m=m+1
                v1=v.swapaxes(0,2)
                # sio.savemat("data/3D_v/v{}.mat".format(m),{"v":v1})
    if i==1:
        v0=vel[87:]
        for j in range(8):
            for k in range(8):
                v=v0[:,100*j:100*j+100,100*k:100*k+100]
                print(v.shape)
                m=m+1
                v1=v.swapaxes(0,2)
                # sio.savemat("data/3D_v/v{}.mat".format(m),{"v":v1})
print(m)
# m=30203
# i=0
# p=np.zeros((200,vel.shape[1],vel.shape[2]))
# p[:,:,:]=vel[0,:,:]
# p[13:,:,:]=vel
# # print(p)
# while (i+600<=801):
#     v=p[:,i:i+600,i:i+600][::2,::6,::6]
#     v=v.swapaxes(0,2)
#     # v=v.swapaxes(0,1)
#     # sio.savemat("data/3D_v_model/v{}.mat".format(m),{"v":v})
#     print(m)
#     i=i+5
#     m=m+1
#     print(v.shape)
# m=29998
# v=p[:,:800,:800][::2,::8,::8]
# v=v.swapaxes(0,2)
# print(v.shape)
# sio.savemat("data/3D_v_model/v{}.mat".format(m),{"v":v})

# plt.figure()
# plt.imshow(v[50,:,:].T)
# plt.colorbar()
# plt.savefig("/home/pengyaoguang/data/3D_net_result/1.png")
    