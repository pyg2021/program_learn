import cigvis
import scipy.io as sio
sx=sio.loadmat("/home/pengyaoguang/data/shengli/data_all_RTM/RTM_easy.mat")["RTM"]
# v=sio.loadmat("/home/pengyaoguang/data/shengli/data_all/fault_v0.mat")["v"]
nodes, cbar = cigvis.create_slices(sx,
                                   cmap='Petrel',
                                   return_cbar=True,
                                   label_str='Amplitude')
nodes.append(cbar)
cigvis.plot3D(nodes, size=(1000, 800), savename='example.png')