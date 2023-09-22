import matplotlib.pyplot as plt
import numpy as np
import math

x_number=6000
y_number=5500

x=np.arange(1,x_number+1,1)
x0=x-x_number/2
# print(x)
d=math.floor(y_number/5.5)
a=(max(x0)*max(x0))/d
b=1/a

# plt.plot(x,[-math.pow(i,2)*b+d+2450 for i in x0])
# plt.plot(x,[-math.pow(i,2)*b+d+2800 for i in x0])
# plt.savefig("test.png")
# plt.show()

y_1=y_number-np.array([math.floor(-math.pow(i,2)*b+y_number*2450/5500) for i in x0])
y_1=y_1+400
y_2=y_number-np.array([math.floor(-math.pow(i,2)*b+y_number*2800/5500) for i in x0])

# y_2=550-[math.floor(-math.pow(i,2)*b+d+280) for i in x0]
print(y_1)
print(y_2)

v_model=np.random.uniform(1.15,1.23,(x_number,y_number))
a=math.floor(1500/5500*y_number)
b=math.floor(4000/5500*y_number)

v_model[:,0:a]=1.5
m=np.linspace(1.5*1.23,1.5*1.23,b)
for j in range(a,y_number):
    v_model[:,j]=m[j-a]
print(v_model.shape)
for i in range(0,x_number):
    for j in range(0,y_number):
        if y_2[i]<=j<=y_1[i]:
            v_model[i,j]=1.2*1.5
plt.figure()
plt.imshow(v_model.T*1000,cmap='seismic')
plt.colorbar()
plt.savefig("program/Ultrasonic/test4.png")
# plt.imsave("program/Ultrasonic/test2.png",v_model.T)

filepath='program/Ultrasonic/v_model4.bin'
binfile = open(filepath, 'wb') #写入
binfile.write(v_model*1000)
binfile.close()


# ny = 2301
# nx = 751
# import torch
# device = torch.device('cuda:1' if torch.cuda.is_available()
#                       else 'cpu')
# v0 = torch.from_file('/mnt/sda/home/yaoguang/deepwave_space/vp.bin',
#                     size=ny*nx).reshape(ny, nx)
# v0=np.array(v0)
# print(v0.shape)
# plt.imshow(v0)
# plt.colorbar()
# plt.imsave("test4.png",v0)