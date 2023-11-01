import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import cv2
import math
from scipy.signal import butter, lfilter
from scipy.io import wavfile
air=sio.loadmat('/home/pengyaoguang/data/Ultrasonic_data/data/air_injection_scan_pre_switch.mat')
real_data_all_test=np.zeros((26,27,6250))
for k in range(26):
    real_data=air['all_traces_pre_switch_CSG'][k].item().T
    #图片的高通滤波
    #傅里叶变换
    img=real_data
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    #设置高通滤波器
    n=1000
    rows, cols = img.shape
    crow,ccol = int(rows/2), int(cols/2)
    mask=np.ones((rows,cols))
    
    mask=np.zeros((rows,cols))
    cen=np.ones_like(mask)
    n=20
    m=50
    m1=100
    mask[crow-m1:crow+m1,ccol-300:ccol+300]=1
    mask[crow-n:crow,ccol-300:ccol+300]=np.repeat(np.linspace(1,0,n),cols,axis=0).reshape(n,cols)
    mask[crow:crow+n,ccol-300:ccol+300]=np.repeat(np.linspace(0,1,n),cols,axis=0).reshape(n,cols)
    mask[crow-m1:crow-m,ccol-300:ccol+300]=np.repeat(np.linspace(0,1,m1-m),cols,axis=0).reshape(m1-m,cols)
    mask[crow+m:crow+m1,ccol-300:ccol+300]=np.repeat(np.linspace(1,0,m1-m),cols,axis=0).reshape(m1-m,cols)
    #傅里叶逆变换
    fshift=fshift*mask
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    # iimg = np.abs(iimg)
    # iimg = iimg.real
    iimg = iimg.real
    real_data_filter=iimg
    a=71
    b=500+500
    for i in range(real_data_filter.shape[1]):
        real_data_filter[:a*i+b,i]=0
    plt.figure()
    max=np.max(real_data_filter[:6250,:])
    plt.imshow(real_data_filter[:6250,:]/max,aspect='auto',cmap='seismic',vmax=1,vmin=-1,extent=(0,real_data_filter.shape[1],0.6,0))
    plt.colorbar()
    plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/data/real_data_process{}.png".format(k))
    plt.close()
    mean_real=np.mean(real_data_filter[:6250,:])
    var_real=var_real=math.sqrt(np.var(real_data_filter[:6250,:]))
    real_data_filter[:6250,:]=(real_data_filter[:6250,:]-mean_real)/var_real
    real_data_all_test[k,k:,:]=real_data_filter[:6250,:].T
real_data_all_test=real_data_all_test[:,:,375:]
sio.savemat("/home/pengyaoguang/data/Ultrasonic_data/data/real_data_all_test.mat",{"data":real_data_all_test[:,:,::5]})