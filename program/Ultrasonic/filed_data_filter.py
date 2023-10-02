import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import cv2
import math
from scipy.signal import butter, lfilter
from scipy.io import wavfile
air=sio.loadmat('/home/pengyaoguang/data/Ultrasonic_data/data/air_injection_scan_pre_switch.mat')
real_data_all_test=np.zeros((26,27,6250))
for k in range(25,26):
    real_data=air['all_traces_pre_switch_CSG'][k].item().T
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
        cen[:,i]=np.linspace(1,0,1000)
    mask[crow-1000:crow, ccol-300:ccol+300] = cen
    cen=np.ones_like(mask[crow:crow+n, ccol-300:ccol+300])
    for i in range(cen.shape[1]):
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
    a=71
    b=500+300
    for i in range(real_data_filter.shape[1]):
        real_data_filter[:a*i+b,i]=0
    plt.figure()
    max=np.max(real_data_filter[:6250,:])
    plt.imshow(real_data_filter[100:6250,:]/max,aspect='auto',cmap='seismic',vmax=0.03,vmin=-0.03,extent=(0,real_data_filter.shape[1],0.6,0))
    plt.colorbar()
    plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/data/real_data_process{}.png".format(k))
    plt.close()
    real_data_all_test[k,k:,:]=real_data_filter[:6250,:].T
mean_real=np.mean(real_data_all_test)
var_real=math.sqrt(np.var(real_data_all_test))
real_data_all_test=(real_data_all_test-mean_real)/var_real
sio.savemat("/home/pengyaoguang/data/Ultrasonic_data/data/real_data_all_test.mat",{"data":real_data_all_test})