import scipy.io as sio
import scipy
vel=sio.loadmat("/home/pengyaoguang/data/3D_v_model/Overthrust_vel.mat")["vel"]/1000
m=30000
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

i=0
while (i+400<=801):
    v=vel[:100,i:i+400,i:i+400][:,::4,::4]
    sio.savemat("data/3D_v_model/v{}.mat".format(m),{"v":v})
    print(m)
    i=i+5
    m=m+1
    