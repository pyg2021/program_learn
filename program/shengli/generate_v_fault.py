import math
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
x=np.arange(100)
y=np.arange(100)
X,Y=np.meshgrid(x,y)
### 三维折叠模型
for epoch in range(5000,10000):
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

    #断层
    d1=random.uniform(5,15)
    d2=random.uniform(5,15)
    a=random.uniform(0,2*math.pi)
    b=random.uniform(1/5*math.pi,1/2*math.pi)
    D1=d1*math.sin(a)+d2*math.cos(a)*math.sin(b)
    D2=d1*math.cos(a)-d2*math.sin(a)*math.cos(b)
    D3=d2*math.sin(b)
    # print(D1,D2,D3)
    c1=math.cos(a)*math.sin(b)
    c2=math.sin(a)*math.sin(b)
    c3=-math.cos(b)
    
    xref=random.uniform(10,90)
    yref=random.uniform(10,90)
    zref=random.uniform(40,60)
    # z1=np.ones((100,100))
    # for i in x:
    #     for j in y:
    #         z1[i][j]=math.cos(2*math.pi*(i 
    zref=0
    Z1=(c1*(X-xref)+c2*(Y-yref))/(-c3)+zref
    Z2=np.ones((100,100))
    d=np.random.uniform(10,15)
    print(c1,c2,c3,np.max(Z1),d)
    for k in range(100)[::-1]:
        for j in range(100):
            for i in range(100):
                if k<Z1[i,j]:
                    d_x=math.floor(i+D1)
                    d_y=math.floor(j+D2)
                    d_z=math.floor(k+D3)
                    # if 0<=d_x<=99:
                    #     if 0<=d_y<=99:
                    #         if 0<=d_z<=99:
                    #             # data[d_x,d_y,d_z]=data[i,j,k]
                    #             # data[d_x,d_y,d_z]=0
                    k_1=math.floor(k+d)
                    if 0<=k_1<=Z1[i,j]:
                        if 0<=k_1<100:
                            data[i,j,k_1]=data[i,j,k]
                        else:
                            data[i,j,99]=data[i,j,k]
                        if k<math.floor(d):
                            data[i,j,k]=data[i,j,0]
    # ax3.plot_surface(x,y,Z1,rstride = 1, cstride = 1,cmap='rainbow')
    scipy.io.savemat("/home/pengyaoguang/data/3D_v_model/v{}.mat".format(epoch), {'v':data})
    # plt.savefig('program/shengli/fault_data/{}.png'.format(epoch))
    # plt.figure()
    # plt.imshow(data[50,:,:].T, aspect='auto', cmap='jet')
    # plt.colorbar(label='km/s')
    # plt.tight_layout()
    # plt.savefig('program/shengli/fault_data/part{}.png'.format(epoch))