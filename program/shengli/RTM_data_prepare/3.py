from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import scipy
import multiprocessing
from multiprocessing import Pool
st=time.time()
vel=sio.loadmat("/home/pengyaoguang/data/3D_v_model/Overthrust_vel.mat")["vel"]/1000
vel=vel.swapaxes(0,2)
# v=vel
# sio.savemat("/home/pengyaoguang/data/3D_v_model/fianl_v.mat",{'v':v})
# points = np.random.rand(v.size, 3)
# grid_x_0, grid_y_0,c_0 = np.mgrid[0:v.shape[0], 0:v.shape[1],0:v.shape[2]]
# points[:,0]=grid_x_0.reshape(-1)
# points[:,1]=grid_y_0.reshape(-1)
# points[:,2]=c_0.reshape(-1)
# # 定义插值点
# grid_x, grid_y,c= np.mgrid[0:v.shape[0],0:v.shape[1] ,0:v.shape[2]:100j]
# # 进行插值
# grid_z = griddata(points, v.reshape(-1), (grid_x, grid_y,c), method='linear')
# sio.save("/home/pengyaoguang/data/3D_v_model/fianl_v.mat",{'v':v})
def run(i):
    print(i)
    v=vel[i]
    # 定义数据点
    points = np.random.rand(v.size, 2)
    grid_x_0, grid_y_0 = np.mgrid[0:v.shape[0], 0:v.shape[1]]
    points[:,0]=grid_x_0.reshape(-1)
    points[:,1]=grid_y_0.reshape(-1)
    # 定义插值点
    grid_x, grid_y= np.mgrid[0:v.shape[0], 0:v.shape[1]:101j]
    # 进行插值
    grid_z = griddata(points, v.reshape(-1), (grid_x, grid_y), method='linear')
    # 绘制插值结果
    # final[i]=grid_z
    # if i%50==0:
    #     plt.imshow(grid_z.T, extent=(0,1,0,1))
    #     plt.colorbar()
    #     # plt.plot(points[:,0], points[:,1], 'o', label='已知点')
    #     # plt.show()
    #     plt.savefig('/home/pengyaoguang/program_learn/program/shengli/RTM_data_prepare/2.png')
    #     plt.close()
    end=time.time()
    print((end-st)/60,'min')
    # sio.savemat("/home/pengyaoguang/data/3D_v_model/fianl_v3.mat",{'v':final})
    return [i,grid_z]
final=np.zeros((801,801,100))
pool = Pool(30)  # 创建拥有3个进程数量的进程池
print(multiprocessing.cpu_count())
# testFL:要处理的数据列表，run：处理testFL列表中数据的函数
res_all=[]
for i in range(vel.shape[0]):
    res=pool.apply_async(run, (i,))
    res_all.append(res)
pool.close()  # 关闭进程池，不再接受新的进程
pool.join()  # 主进程阻塞等待子进程的退出
print(len(res_all))
for y in res_all:
    i,x=y.get()
    final[i]=x[:,:100]
sio.savemat("/home/pengyaoguang/data/3D_v_model/fianl_v3.mat",{'v':final})