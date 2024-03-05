import scipy.io as sio
vel=sio.loadmat("data/3D_v/Overthrust_vel.mat")["vel"]
m=20000
for i in range(2):
    if i==0:
        v0=vel[100*i:100*i+100]
        for j in range(8):
            v=v0[:,100*j:100*j+100,100*j:100*j+100]
            print(v.shape)
            m=m+1
            sio.savemat("data/3D_v/v{}.mat".format(m),{"v":v})
    if i==1:
        v0=vel[87:]
        for j in range(8):
            v=v0[:,100*j:100*j+100,100*j:100*j+100]
            print(v.shape)
            m=m+1
            sio.savemat("data/3D_v/v{}.mat".format(m),{"v":v})