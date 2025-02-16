import numpy as np
import matplotlib.pyplot as plt
# data=np.fromfile('/home/data/imp.npy')
data=np.load('/home/data/imp.npy')
for i in range(502):
    if i%10==0:
        # plt.figure()
        plt.imshow(data[::4,i,::5])
        plt.colorbar()
        plt.savefig('/home/pengyaoguang/program_learn/2D/0.png')
        plt.close()
        print(data[::4,i,::5].shape)
        break
    # print(data[:,:,50])
