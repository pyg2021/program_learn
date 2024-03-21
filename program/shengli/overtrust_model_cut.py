import scipy.io as sio
import scipy
import numpy as np
import matplotlib.pyplot as plt
vel=sio.loadmat("/home/pengyaoguang/data/3D_v_model/Overthrust_vel.mat")["vel"]/1000

print(vel.shape)
# for i in range(2):
#     if i==0:
#         v0=vel[100*i:100*i+100]
#         for j in range(8):
#             for k in range(8):
#                 v=v0[:,100*j:100*j+100,100*k:100*k+100]
#                 print(v.shape)
#                 m=m+1
#                 v1=v.swapaxes(0,2)
#                 sio.savemat("data/3D_v/v{}.mat".format(m),{"v":v1})
#     if i==1:
#         v0=vel[87:]
#         for j in range(8):
#             for k in range(8):
#                 v=v0[:,100*j:100*j+100,100*k:100*k+100]
#                 print(v.shape)
#                 m=m+1
#                 v1=v.swapaxes(0,2)
#                 sio.savemat("data/3D_v/v{}.mat".format(m),{"v":v1})
m=30000
i=0
p=np.zeros((100,100,100))
while (i+400<=801):
    v=vel[:100,i:i+400,i:i+400][::2,::4,::4]
    v=v.swapaxes(0,2)
    sio.savemat("data/3D_v_model/v{}.mat".format(m),{"v":v})
    print(m)
    i=i+5
    m=m+1
m=29999
v=vel[:100,:800,:800][:,::8,::8]
v=v.swapaxes(0,2)
print(v.shape)
sio.savemat("data/3D_v_model/v{}.mat".format(m),{"v":v})

# plt.figure()
# plt.imshow(v[50,:,:].T)
# plt.colorbar()
# plt.savefig("/home/pengyaoguang/data/3D_net_result/1.png")
    