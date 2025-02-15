import segyio
import numpy as np
import matplotlib.pyplot as plt
filename='/home/data/well_data_depth.mat'
import scipy.io as sio
data=sio.loadmat(filename)
print(data.keys())
print(len(data['well_depth_velocity'][0]))
print(len(data['well_depth_velocity'][0][0][0]))
print(data['well_h_w_tstart_len'])