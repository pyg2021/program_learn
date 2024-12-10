##update 2024.12.09  生成二维数据
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
import time
def floded():
    x=np.arange(100)
    # y=np.arange(100)
    # X,Y=np.meshgrid(x,y)
    z=0
    data_10=np.ones((10,100))*101
    n=1
    for i in range(100):
        # produce layer
        T=random.uniform(50,150)
        # a=random.uniform(-1,1)
        A=random.uniform(3,5)
        dz=random.uniform(10,25)
        z=z+dz
        Z1=np.cos(2*math.pi*(x)/T)*A+z
        data_10[i,:]=Z1
        n=n+1
        # ax3.plot_surface(x,y,Z1,rstride = 1, cstride = 1,cmap='rainbow')
        if z>75:
            break
    # data_10=np.floor(data_10)
    data=np.ones((100,100))
    v=random.uniform(1.8,2)
    for i in range(n):
        for j in range(data_10.shape[1]):
                if i==0:
                    data[j,:math.floor(data_10[i,j])]=v
                else:
                    data[j][math.floor(data_10[i-1,j]):math.floor(data_10[i,j])]=v
        v=v+random.uniform(0.4,0.8)
    # v=v+random.uniform(1,1.5)
    data[j,math.floor(data_10[i,j]):]=v
    return data
def salt(data0):
    v=np.max(data0)
    ## 盐丘模型
    data=data0
    a=random.uniform(50,100)
    xref=random.uniform(20,80)
    # yref=random.uniform(20,80)
    # xref=50
    # yref=50
    # sx=random.uniform(7.4,18)
    # sy=random.uniform(7.4,18)
    # sx=15
    # sy=7.4
    # print(sx,sy)
    # p=random.uniform(0.1,1.9)*math.pi
    # d1=math.pow(math.cos(p),2)/2/math.pow(sx,2)+math.pow(math.sin(p),2)/2/math.pow(sy,2)
    # # print(d1)
    # d2=-math.pow(2*math.sin(p),1)/4/math.pow(sx,2)+math.pow(2*math.sin(p),1)/4/math.pow(sy,2)
    # d3=math.pow(math.sin(p),2)/2/math.pow(sx,2)+math.pow(math.cos(p),2)/2/math.pow(sy,2)
    # print(sx,sy,p/math.pi)
    d1=random.uniform(20,200)
    Z2=np.ones((100))
    for i in range(100):
        # for j in range(100):
            Z2[i]=a*math.exp(-(0.5*math.pow((i-xref),2))/d1)
    Z2=101-Z2
    v1=random.uniform(0.6,1)
    for i in range(Z2.shape[0]):
            data[i,math.floor(Z2[i]):]=v+v1
    # np.save("program/shengli/v.bin",data)
    if np.min(Z2)<0:
        return salt(data0)
    return data

def fault(data):
    #断层
    x=np.arange(100)
    y=np.arange(100)
    X,Y=np.meshgrid(x,y)
    d1=random.uniform(5,15)
    d2=random.uniform(5,15)
    a=random.uniform(1/8*math.pi,7/8*math.pi)
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
    Z1=math.tan(a)*(x-xref)
    Z2=np.ones((100,100))
    d=np.random.uniform(10,15)
    for k in range(100)[::-1]:
            for i in range(100):
                if k<Z1[i]:
                    # if 0<=d_x<=99:
                    #     if 0<=d_y<=99:
                    #         if 0<=d_z<=99:
                    #             # data[d_x,d_y,d_z]=data[i,j,k]
                    #             # data[d_x,d_y,d_z]=0
                    k_1=math.floor(k+d)
                    if 0<=k_1<=Z1[i]:
                        if 0<=k_1<100:
                            data[i,k_1]=data[i,k]
                        else:
                            data[i,99]=data[i,k]
                        if k<math.floor(d):
                            data[i,k]=data[i,0]
    return data
for epoch in range(20000,30000):
    data=floded()
    data=fault(data)
    data=salt(data)
    data=fault(data)

    # # data=fault(data)
    # scipy.io.savemat("/home/pengyaoguang/data/2D_data/2D_v_model1209/v{}.mat".format(epoch), {'v':data})
    data.tofile('/home/pengyaoguang/data/2D_data/2D_v_model1209/v{}.bin'.format(epoch))


    #figure
    if epoch%100==0:
        plt.figure()
        plt.imshow(data.T)
        plt.colorbar()
        plt.savefig("/home/pengyaoguang/program_learn/2D/2d_Rtm_data/1.png")
        plt.close()
    print(epoch)
    # time.sleep(1)