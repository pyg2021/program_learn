# Determine the size and x and z
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import os
size=5
x=201
z=101
number=100
data_number=np.zeros((number,x,z))
for k in range(number):
    data_all=np.zeros((x,z))
    x_number=math.floor(x/size)
    z_number=math.floor(z/size)
    for i in range(x_number):
        for j in range(z_number):
            # np.random.seed()
            data_all[i*size:i*size+size,j*size:j*size+size]=random.uniform(1.5,6)
    for j in range(z_number):
        data_all[x_number*size:x,j*size:j*size+size]=random.uniform(1.5,6)
    for i in range(x_number):
        data_all[i*size:i*size+size,z_number*size:z]=random.uniform(1.5,6)
    data_all[x_number*size:x,z_number*size:z]=random.uniform(1.5,6)
    print(data_all.shape)
#veiw and save 
    plt.imshow(data_all)
    plt.colorbar()
    plt.imsave("program/shengli/data_result2/{}.png".format(k),data_all.T)
    filepath='program/shengli/data_result2/v_model{}.bin'.format(k)
    binfile = open(filepath, 'wb') #写入
    binfile.write(data_all)
    # print('content',data_all)
    binfile.close()
    data_number[k,:,:]=data_all
## save data
# binfile = open('program/shengli/data_result2/v_model_3d.bin', 'wb') #写入
# binfile.write(data_number)
# # print('content',data_all)
# binfile.close()
from scipy.io import savemat
savemat('program/shengli/data_result2/model_3d.mat',{'V':data_number})


readfile=open(filepath,"rb")
data=np.fromfile(filepath).reshape(x,z)
# print(data.shape)
binfile.close()


