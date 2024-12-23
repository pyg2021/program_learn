import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

#定义三维数据
xx = np.arange(-5,5,0.5)
yy = np.arange(-5,5,0.5)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(X)+np.cos(Y)


#作图
ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow')
# ax3.contour(X,Y,Z, zdim='z',offset=-2,cmap='rainbow')   
#等高线图，要设置offset，为Z的最小值
# plt.show()
plt.savefig('program/shengli/figure.png')