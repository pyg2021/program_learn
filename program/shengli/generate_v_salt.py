import math
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
import time
start=time.time()
x=np.arange(100)
y=np.arange(100)
X,Y=np.meshgrid(x,y)
epoch=0### 三维折叠模型
while epoch<5000:
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
    
    ## 盐丘模型
    a=random.uniform(10,30)
    xref=random.uniform(0,100)
    yref=random.uniform(0,100)
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
    # a=11.71479138458772
    # d1=0.010813124017268185
    # # print(d1)
    # d2=0.008611597972634052
    # d3=0.0025399845017029937
    # xref=23.905875829394713
    # yref=34.24228348218066
    # 11.71479138458772 0.010813124017268185 -0.008611597972634052 0.0025399845017029937 
    # 23.905875829394713 34.24228348218066
    Z2=np.ones((100,100))
    for i in range(100):
        for j in range(100):
            Z2[i,j]=a*math.exp(-(d1*math.pow((i-xref),2)+d3*math.pow((j-yref),2)+2*d2*(i-xref)*(j-yref)))
            # if Z2[i,j]>=100:
            #     Z2[i,j]=0
    Z2=101-Z2
    # if np.min(Z2)>20:
    #     continue
    print(np.min(Z2),epoch,p/math.pi)
    # Z2=a*np.exp(-(d1*np.power((x-xref),2)+d3*np.power((y-yref),2)+2*d2*(x-xref).reshape(1,100)*(y-yref).reshape(100,1)))
    # ax3.plot_surface(x,y,Z2,rstride = 1, cstride = 1,cmap='rainbow')
    v1=random.uniform(0.6,1)
    for i in range(Z2.shape[0]):
        for j in range(Z2.shape[1]):
            data[i,j,math.floor(Z2[i,j]):]=v+v1
    # np.save("program/shengli/v.bin",data)
    if np.min(Z2)<0:
        continue
    scipy.io.savemat("/home/pengyaoguang/work_space/program/shengli/data_all/salt_v{}.mat".format(epoch), {'v':data})



    # z1=np.ones((100,100))
    # for i in x:
    #     for j in y:
    #         z1[i][j]=math.cos(2*math.pi*(i+1*j)/200)*5+10
    
    # Z2=np.cos(2*math.pi*(X+1*Y)/200)*5
    # ax3.plot_surface(x,y,Z2,rstride = 1, cstride = 1,cmap='rainbow')
    # plt.show()
    # plt.savefig('program/shengli/salt_data/{}.png'.format(epoch))
    # plt.figure()
    # plt.imshow(data[50,:,:].T, aspect='auto', cmap='jet')
    # plt.colorbar(label='km/s')
    # plt.tight_layout()
    # plt.savefig('program/shengli/salt_data/part{}.png'.format(epoch))
    # print(1)
    epoch+=1
end=time.time()
print(end-start,'s')