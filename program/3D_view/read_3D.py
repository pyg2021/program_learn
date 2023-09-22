import scipy.io as scio
from mayavi import mlab
import numpy as np
import hdf5storage

path='program/shengli/data_result2/v_model_3d.mat'
# data = scio.loadmat(path)
data = hdf5storage.loadmat(path)
#查看字典的键
print(data.keys())
cube = data['V']
# 查看数据维度
print(cube.shape)
# print(cube[:])
# 原数据的维度为(300*1066*60) ，交换日号和2号维度，变成(6日，6日，300)
cube1 = cube.swapaxes(0,2)
print(cube1.shape)

# cube1 = cube1/np.max(cube1)
# cube2 = np.zeros_like(cube3)# for i in range(cube.shape[0]-1，-1):社cube2[275-i,:,:]= cube1[i,:,:]
#
# cube1 = cube2print(np.max(cube1))# cube1 = cube-cube2
clip = 1
# vmin, vmax = -clip, clip
vmin, vmax = np.min(cube1), np.max(cube1)
print(cube1.shape)
source = mlab.pipeline.scalar_field(cube1) # scalar_field获得数据的标量数据场source.spacing = [1，1，2] #数据间隔

source.spacing = [1,1,2] #数据间隔
mlab.figure(figure=1,bgcolor=0,fgcolor=None,size=(600,600))

mlab.pipeline.image_plane_widget(source,
plane_orientation='x_axes', ##设置切平面的方向
slice_index=0,colormap='hsv',vmin=vmin,vmax=vmax)

mlab.pipeline.image_plane_widget(source,
plane_orientation='y_axes', ##设置切平面的方向
slice_index=0,colormap='hsv',vmin=vmin,vmax=vmax)

mlab.pipeline.image_plane_widget(source,
plane_orientation='z_axes', ##设置切平面的方向
slice_index=0,colormap='hsv',vmin=vmin,vmax=vmax)

mlab.pipeline.image_plane_widget(source,
plane_orientation='x_axes', ##设置切平面的方向
slice_index=2000,colormap='hsv',vmin=vmin,vmax=vmax)

mlab.pipeline.image_plane_widget(source,
plane_orientation='y_axes', ##设置切平面的方向
slice_index=2000,colormap='hsv',vmin=vmin,vmax=vmax)

mlab.pipeline.image_plane_widget(source,
plane_orientation='z_axes', ##设置切平面的方向
slice_index=2000,colormap='hsv',vmin=vmin,vmax=vmax)

mlab.colorbar(title='Simple')
mlab.show()
mlab.savefig('1.png')