import scipy.io as sio
vel=sio.loadmat("data/3D_v/Overthrust_vel.mat")["vel"]/1000
m=20000
for i in range(2):
    if i==0:
        v0=vel[100*i:100*i+100]
        for j in range(8):
            for k in range(8):
                v=v0[:,100*j:100*j+100,100*k:100*k+100]
                print(v.shape)
                m=m+1
                v1=v.swapaxes(0,2)
                sio.savemat("data/3D_v/v{}.mat".format(m),{"v":v1})
    if i==1:
        v0=vel[87:]
        for j in range(8):
            for k in range(8):
                v=v0[:,100*j:100*j+100,100*k:100*k+100]
                print(v.shape)
                m=m+1
                v1=v.swapaxes(0,2)
                sio.savemat("data/3D_v/v{}.mat".format(m),{"v":v1})