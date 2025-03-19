##update 2024.03.05
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
import time

def re_f(n,c,X,Y):
    if n==0:
        return c
    T=random.uniform(50,150)
    a=random.uniform(-1,1)
    A=random.uniform(1,3)
    return re_f(n-1,c+np.cos(2*math.pi*(X+a*Y)/T)*A,X,Y)
def floded(num):
    x=np.arange(100)
    y=np.arange(100)
    X,Y=np.meshgrid(x,y)
    z=0
    data_10=np.ones((10,100,100))*101
    n=1
    for i in range(100):
        # produce layer
        nn=10
        # T=random.uniform(50,150)
        # a=random.uniform(-1,1)
        # A1=3
        # A2=5
        # A=random.uniform(3,5)
        dz=random.uniform(10,25)
        T=random.uniform(50,150)
        a=random.uniform(-1,1)
        A=random.uniform(3,5)
        # cc=np.cos(2*math.pi*(X+a*Y)/T)*A
        # for _ in range(nn):
        #     T=random.uniform(50,150)
        #     a=random.uniform(-1,1)
        #     A=random.uniform(A1,A2)
        #     dz=random.uniform(10,25)
        #     cc=cc+np.cos(2*math.pi*(X+a*Y)/T)*A
        c=np.cos(2*math.pi*(X+a*Y)/T)*A
        cc=re_f(num,c,X,Y)
        z=z+dz
        print(z)
        # print(np.max(cc))
        # Z1=np.cos(2*math.pi*(X+a*Y)/T)*A+z
        Z1=cc+z
        data_10[i,:,:]=Z1
        n=n+1
        # ax3=plt.figure(6,6)
        fig = plt.figure(figsize=(12,6))
        ax3 = fig.add_subplot(121, projection='3d')
        ax3.plot_surface(x,y,Z1,rstride = 1, cstride = 1,cmap='rainbow')
        plt.savefig("/home/pengyaoguang/program_learn/new_g_v_model/2.png")
        # plt.savefig
        if z>75:
            break
    # data_10=np.floor(data_10)
    
    data=np.ones((100,100,100))
    v=random.uniform(1.8,2)
    # print(n)
    for i in range(n):
        
        for j in range(data_10.shape[1]):
            for k in range(data_10.shape[2]):
                if i==0:
                    data[j,k,:math.floor(data_10[i,j,k])]=v
                else:
                    if math.floor(data_10[i-1,j,k])<=math.floor(data_10[i,j,k]):
                        data[j][k][math.floor(data_10[i-1,j,k]):math.floor(data_10[i,j,k])]=v
                    else:
                        data[j][k][math.floor(data_10[i,j,k]):math.floor(data_10[i-1,j,k])]=v
                    # print(math.floor(data_10[i-1,j,k]),math.floor(data_10[i,j,k]))
        # print('0000000000000000000')
        v=v+random.uniform(0.4,0.8)
    # v=v+random.uniform(1,1.5)
    data[j,k,math.floor(data_10[i,j,k]):]=v
    return data
def salt(data0):
    v=np.max(data0)
    ## 盐丘模型
    data=data0
    a=random.uniform(50,100)
    xref=random.uniform(20,80)
    yref=random.uniform(20,80)
    # xref=50
    # yref=50
    sx=random.uniform(7.4,18)
    sy=random.uniform(7.4,18)
    # sx=15
    # sy=7.4
    # print(sx,sy)
    p=random.uniform(0.1,1.9)*math.pi
    d1=math.pow(math.cos(p),2)/2/math.pow(sx,2)+math.pow(math.sin(p),2)/2/math.pow(sy,2)
    # print(d1)
    d2=-math.pow(2*math.sin(p),1)/4/math.pow(sx,2)+math.pow(2*math.sin(p),1)/4/math.pow(sy,2)
    d3=math.pow(math.sin(p),2)/2/math.pow(sx,2)+math.pow(math.cos(p),2)/2/math.pow(sy,2)
    print(sx,sy,p/math.pi)
    Z2=np.ones((100,100))
    for i in range(100):
        for j in range(100):
            Z2[i,j]=a*math.exp(-(d1*math.pow((i-xref),2)+d3*math.pow((j-yref),2)+2*d2*(i-xref)*(j-yref)))
    Z2=101-Z2
    if np.min(Z2)<0:
        return salt(data0)
    top=np.min(Z2)
    d=0
    # while True:
    #     if top<21:
    #         break
    #     d=int(random.uniform(20,50))
    #     if d<top:
    #         break
    # v1=random.uniform(0.6,1)
    v1=random.uniform(1.5,2)
    for i in range(Z2.shape[0]):
        for j in range(Z2.shape[1]):
            # print(math.floor(Z2[i,j]-d))
            data[i,j,math.floor(Z2[i,j]-d):101-d-1]=v+v1
    # np.save("program/shengli/v.bin",data)
    
    
    return data

def fault(data):
    #断层
    x=np.arange(100)
    y=np.arange(100)
    X,Y=np.meshgrid(x,y)
    d1=random.uniform(5,15)
    d2=random.uniform(5,15)
    a=random.uniform(1/8*math.pi,2*math.pi)
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
    return data
# for epoch in range(25300,25500):
data=floded(3)
data=salt(data)
data=fault(data)

data=fault(data)

# data=fault(data)
# scipy.io.savemat("/home/pengyaoguang/data/3D_v_model/v{}.mat".format(epoch), {'v':data})

import scipy.io as sio
# sio.savemat('/home/pengyaoguang/data/3D_fuse/floded_2.mat',{'v':data})
# sio.savemat('/home/pengyaoguang/data/3D_fuse/fault.mat',{'v':data})
# sio.savemat('/home/pengyaoguang/data/3D_fuse/salt.mat',{'v':data})
sio.savemat('/home/pengyaoguang/data/3D_fuse/all.mat',{'v':data})
#figure
plt.figure()
plt.imshow(data[50].T,cmap='jet')
plt.colorbar()
# plt.savefig('/home/pengyaoguang/data/3D_fuse/floded_2.eps',dpi=300)
# plt.savefig('/home/pengyaoguang/data/3D_fuse/fault.eps',dpi=300)
# plt.savefig('/home/pengyaoguang/data/3D_fuse/salt.eps',dpi=300)
plt.savefig('/home/pengyaoguang/data/3D_fuse/all.eps',dpi=300)
plt.savefig("/home/pengyaoguang/program_learn/new_g_v_model/1.png")
# break