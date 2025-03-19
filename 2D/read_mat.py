import segyio
import numpy as np
import matplotlib.pyplot as plt
# filename='/home/data/well_data_depth.mat'
# filename='/home/pengyaoguang/data/3D_v_model/Overthrust_vel.mat'

import scipy.io as sio
import time
# data=sio.loadmat(filename)
# print(data['vel'].shape)
##25、80
k=40
# res=[25,40,90]
res=[85]
j=50
for k in res:
    # time.sleep(2)
    data=sio.loadmat("/home/pengyaoguang/data/3D_RTM2/v{}".format(k-1))["v"]*1000
    plt.figure()
    plt.imshow(data[j].T)
    plt.colorbar()
    plt.savefig("/home/pengyaoguang/program_learn/2D/0.png")
    plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/{}_{}v_real_test.eps".format(k,j),dpi=300)
# print(data.keys())
# print(len(data['well_depth_velocity'][0]))
# print(len(data['well_depth_velocity'][0][0][0]))
# print(data['well_h_w_tstart_len'])