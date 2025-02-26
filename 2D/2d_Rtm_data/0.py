import scipy.io as sio
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
k=25102
v=sio.loadmat("/home/pengyaoguang/data/3D_v_model/v{}".format(k))["v"]*1000
plt.imshow(v[50].T)
plt.savefig('/home/pengyaoguang/program_learn/2D/2d_Rtm_data/1.png')