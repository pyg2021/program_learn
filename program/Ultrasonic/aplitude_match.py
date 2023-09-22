import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import cv2
import math
from scipy.signal import butter, lfilter
from scipy.io import wavfile
air=sio.loadmat('program/Ultrasonic/data/air_injection_scan_pre_switch.mat')
print()
real_data=air['all_traces_pre_switch_CSG'][0].item().T
#图片的高通滤波
#傅里叶变换
img=real_data
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
#设置高通滤波器
n=1000
rows, cols = img.shape
mask=np.ones((rows,cols))
crow,ccol = int(rows/2), int(cols/2)
cen=np.ones_like(mask[crow-1000:crow, ccol-300:ccol+300])
for i in range(cen.shape[1]):
    cen[:,i]=np.linspace(1,0,n)
mask[crow-1000:crow, ccol-300:ccol+300] = cen
cen=np.ones((n,27))
for i in range(27):
    cen[:,i]=np.linspace(0,1,n)
mask[crow:crow+n, ccol-300:ccol+300] = cen

cen=np.ones_like(mask[crow+700:crow+1800, ccol-300:ccol+300])
for i in range(cen.shape[1]):
    cen[:,i]=np.linspace(1,0,cen.shape[0])
mask[crow+700:crow+1800, ccol-300:ccol+300] = cen
cen=np.ones_like(mask[crow-1800:crow-900, ccol-300:ccol+300])
for i in range(cen.shape[1]):
    cen[:,i]=np.linspace(0,1,cen.shape[0])
mask[crow-1800:crow-900, ccol-300:ccol+300] = cen

mask[crow-300:crow, ccol-300:ccol+300] = 0
mask[crow:crow+300, ccol-300:ccol+300] = 0
mask[crow+1800:, ccol-300:ccol+300] = 0
mask[:crow-1800, ccol-300:ccol+300] = 0
#傅里叶逆变换
fshift=fshift*mask
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
# iimg = np.abs(iimg)
# iimg = iimg.real
iimg = iimg.real
real_data_filter=iimg
synthetic=sio.loadmat('program/Ultrasonic/data4.mat')
synthetic_data=synthetic['V'][0].T
max=np.max(real_data_filter[:6250,:])
plt.imshow(real_data_filter[0:6250,:]/max,aspect='auto',cmap='seismic',vmax=0.03,vmin=-0.03,extent=(0,26,0.564,0))
plt.colorbar()
plt.savefig("program/Ultrasonic//result/aplitude_real_data_filter.png")
a=65
b=1200
for i in range(real_data_filter.shape[1]):
    real_data_filter[:a*i+b,i]=0
plt.figure()
max=np.max(real_data_filter[:6250,:])
mean_real=np.mean(real_data_filter)
var_real=math.sqrt(np.var(real_data_filter))
print(max)
# plt.imshow((real_data_filter[0:6250,:]-mean_real)/var_real,aspect='auto',cmap='seismic',vmax=0.03,vmin=-0.03,extent=(0,26,0.6,0))
plt.imshow((real_data_filter[:6250,:]-mean_real)/var_real,aspect='auto',cmap='seismic',vmax=1,vmin=-1,extent=(0,26,0.564,0))
plt.colorbar()
plt.savefig("program/Ultrasonic//result/aplitude_real_data_filter_cut.png")

plt.figure()
max=np.max(synthetic_data[:25000,])
plt.imshow(synthetic_data[:25000,]/max,aspect='auto',cmap='seismic',vmax=0.005,vmin=-0.005,extent=(0,26,0.564,0))
plt.colorbar()
plt.savefig("program/Ultrasonic//result/synthetic_data.png")
a=300
b=3500
for i in range(synthetic_data.shape[1]):
    synthetic_data[:a*i+b,i]=0
plt.figure()
max=np.max(synthetic_data[:25000,])
print(max)
mean_real=np.mean(synthetic_data)
var_real=math.sqrt(np.var(synthetic_data))
plt.imshow((synthetic_data[:25000,]-mean_real)/var_real,aspect='auto',cmap='seismic',vmax=1,vmin=-1,extent=(0,26,0.564,0))
# plt.imshow((synthetic_data[:25000,]-mean_real)/var_real,aspect='auto',cmap='seismic',vmax=0.03,vmin=-0.03,extent=(0,26,0.564,0))
plt.colorbar()
plt.savefig("program/Ultrasonic//result/synthetic_data_cut.png")



