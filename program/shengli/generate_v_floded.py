import math
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
x=np.arange(100)
y=np.arange(100)
X,Y=np.meshgrid(x,y)
### 三维折叠模型
for epoch in range(5000):
    z=0
    # plt.figure()
    ax3 = plt.axes(projection='3d')
    data_10=np.ones((10,100,100))*101
    n=1
    for i in range(100):
        # produce layer
        T=random.uniform(150,200)
        a=random.uniform(-1,1)
        A=random.uniform(3,5)
        dz=random.uniform(14,20)
        z=z+dz
        Z1=np.cos(2*math.pi*(X+a*Y)/T)*A+z
        data_10[i,:,:]=Z1
        n=n+1
        # ax3.plot_surface(x,y,Z1,rstride = 1, cstride = 1,cmap='rainbow')
        if z>75:
            break
    # data_10=np.floor(data_10)
    data=np.ones((100,100,100))
    v=random.uniform(1.8,2)
    for i in range(n):
        
        for j in range(data_10.shape[1]):
            for k in range(data_10.shape[2]):
                if i==0:
                    data[j,k,:math.floor(data_10[i,j,k])]=v
                else:
                    data[j][k][math.floor(data_10[i-1,j,k]):math.floor(data_10[i,j,k])]=v
        v=v+random.uniform(0.4,0.8)
    # v=v+random.uniform(1,1.5)
    data[j,k,math.floor(data_10[i,j,k]):]=v
    scipy.io.savemat("/home/pengyaoguang/data/3D_v_model/v{}.mat".format(epoch), {'v':data})
    # plt.show()
    # plt.savefig('program/shengli/floded_data/{}.png'.format(epoch))
    # plt.figure()
    # plt.imshow(data[50,:,:].T, aspect='auto', cmap='jet')
    # plt.colorbar(label='km/s')
    # plt.tight_layout()
    # plt.savefig('program/shengli/floded_data/part{}.png'.format(epoch))