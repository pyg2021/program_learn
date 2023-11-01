# Determine the size and number
import numpy as np
import random
import matplotlib.pyplot as plt
import os
for k in range(10):
    size=5
    x=1000
    z=500
    data_all=np.zeros((x*size,z*size))
    for i in range(x):
        for j in range(z):
            # np.random.seed()
            data_all[i*size:i*size+size,j*size:j*size+size]=random.uniform(1.5,6)

    print(data_all.shape)
#veiw and save 
    plt.imshow(data_all)
    plt.imsave("/home/pengyaoguang/deepwave_space/program/shengli/data_result1/{}.png".format(k),data_all.T)
    filepath='/home/pengyaoguang/deepwave_space/program/shengli/data_result1/v_model{}.bin'.format(k)
    binfile = open(filepath, 'wb') #写入
    binfile.write(data_all)
    # print('content',data_all)
    binfile.close()
    
readfile=open(filepath,"rb")
data=np.fromfile(filepath).reshape(x*size,z*size)
print(data.shape)
binfile.close()


