import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import cv2
air=sio.loadmat('/home/pengyaoguang/data/Ultrasonic_data/data/air_injection_scan_pre_switch.mat')
real_data=air['all_traces_pre_switch_CSG'][0].item().T

# #高通滤波
# from scipy.signal import butter, lfilter
# from scipy.io import wavfile
# b, a = butter(4, 0.1, btype="highpass")
# real_1=real_data
# # real_1[:160,:1]=0
# real_data_filter= lfilter(b, a,real_1)
# t=np.arange(0,10006)
# plt.plot(t,real_data_filter,label='filter')
# plt.plot(t,real_1,label='real')
# plt.show()
# plt.legend()
# plt.savefig('1.png')
# print()
#高通滤波
from scipy.signal import butter, lfilter
from scipy.io import wavfile
#图片的高通滤波
#傅里叶变换
img=real_data
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
#设置高通滤波器
n=1000
rows, cols = img.shape
crow,ccol = int(rows/2), int(cols/2)
##set mask
# mask=np.ones((rows,cols))

# cen=np.ones_like(mask[crow-1000:crow, ccol-300:ccol+300])
# for i in range(cen.shape[1]):
#     cen[:,i]=np.linspace(1,0,n)
# mask[crow-1000:crow, ccol-300:ccol+300] = cen
# cen=np.ones((n,27))
# for i in range(27):
#     cen[:,i]=np.linspace(0,1,n)
# mask[crow:crow+n, ccol-300:ccol+300] = cen

# cen=np.ones_like(mask[crow+700:crow+1800, ccol-300:ccol+300])
# for i in range(cen.shape[1]):
#     cen[:,i]=np.linspace(1,0,cen.shape[0])
# mask[crow+700:crow+1800, ccol-300:ccol+300] = cen
# cen=np.ones_like(mask[crow-1800:crow-900, ccol-300:ccol+300])
# for i in range(cen.shape[1]):
#     cen[:,i]=np.linspace(0,1,cen.shape[0])
# mask[crow-1800:crow-900, ccol-300:ccol+300] = cen

# mask[crow-300:crow, ccol-300:ccol+300] = 0
# mask[crow:crow+300, ccol-300:ccol+300] = 0
# mask[crow+1800:, ccol-300:ccol+300] = 0
# mask[:crow-1800, ccol-300:ccol+300] = 0
mask=np.zeros((rows,cols))
cen=np.ones_like(mask)
n=20
m=50
m1=100
mask[crow-m1:crow+m1,ccol-300:ccol+300]=1
mask[crow-n:crow,ccol-300:ccol+300]=np.repeat(np.linspace(1,0,n),27,axis=0).reshape(n,27)
mask[crow:crow+n,ccol-300:ccol+300]=np.repeat(np.linspace(0,1,n),27,axis=0).reshape(n,27)
mask[crow-m1:crow-m,ccol-300:ccol+300]=np.repeat(np.linspace(0,1,m1-m),27,axis=0).reshape(m1-m,27)
mask[crow+m:crow+m1,ccol-300:ccol+300]=np.repeat(np.linspace(1,0,m1-m),27,axis=0).reshape(m1-m,27)
#傅里叶逆变换
fshift=fshift*mask
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
# iimg = np.abs(iimg)
# iimg = iimg.real
iimg = iimg.real
real_data_filter=iimg



synthetic=sio.loadmat('/home/pengyaoguang/data/Ultrasonic_data/test10/synthetic_data0.mat')
synthetic_data=synthetic['V'][0].T

plt.figure(figsize=(20,20))
max=np.max(synthetic_data[:,])
plt.imshow(synthetic_data[:,]/max,aspect='auto',cmap='seismic',vmax=0.5,vmin=-0.5,extent=(0,26,0.564,0))
plt.colorbar()
# plt.colorbar()
# plt.xlabel("Channel")
# plt.ylabel("Time Sample(ms)")
# plt.set_cmap("gray")
plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/test10/synthetic_data.png")
plt.figure()
max=np.max(real_data[:6250,:])
plt.imshow(real_data[:6250,:]/max,aspect='auto',cmap='seismic',vmax=1,vmin=-1,extent=(0,26,0.6,0))
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/test10/real_data.png")
plt.figure()
max=np.max(real_data_filter[:6250,:])
plt.imshow(real_data_filter[:6250,:]/max,aspect='auto',cmap='seismic',vmax=1,vmin=-1,extent=(0,26,0.6,0))
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/test10/real_data_filter.png")
shot_1_filter=sio.loadmat('/home/pengyaoguang/1325/program/Ultrasonic/shot_1_filter.mat')
shot_1_filter=shot_1_filter['sht_a_f']
plt.figure()
max=np.max(shot_1_filter[:6250,:])
plt.imshow(shot_1_filter[:6250,:]/max,aspect='auto',cmap='seismic',vmax=1,vmin=-1,extent=(0,26,0.6,0))
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/test10/shot_1_filter.png")

## synthetic data fft
fft_data_new=np.zeros((1175))
from scipy.fftpack import fft,fftshift
for i in range(27):
    N=2350
    fft_data = fft(synthetic_data[:2350,i])
    fft_amp0 = np.array(np.abs(fft_data)/N*2)   # 用于计算双边谱
    fft_amp0[0]=0.5*fft_amp0[0]
    N_2 = int(N/2)
    fft_amp1 = fft_amp0[0:N_2]  # 单边谱
    # 计算频谱的频率轴
    list0 = np.array(range(0, N))
    list1 = np.array(range(0, int(N/2)))
    list0_shift = np.array(range(0, N))
    dt=2.4*1e-7
    sample_freq=1/dt
    freq0 = sample_freq*list0/N        # 双边谱的频率轴
    freq1 = sample_freq*list1/N        # 单边谱的频率轴
    # # 单边谱
    fft_data_new=fft_data_new+fft_amp1
fft_amp1=fft_data_new
plt.figure()
nmax=np.max(fft_amp1[:])
plt.plot(freq1[:], fft_amp1[:]/nmax,label='synthetic_data')
plt.legend()
plt.title(' spectrum single-sided')
plt.xlabel('frequency  (Hz)')
plt.ylabel(' Amplitude ')
plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/test10/spectrum_synthetic_data.png")


## spectrum_real_data
# water=sio.loadmat('/home/pengyaoguang/data/Ultrasonic_data/data/air_injection_scan_pre_switch.mat')
# real_data=water['all_traces_pre_switch_CSG'][0].item().T
# fft_data_new=np.zeros((5003))
# for i in range(27):
#     N=10006
#     fft_data = fft(real_data[:10006,i])
#     fft_amp0 = np.array(np.abs(fft_data)/N*2)   # 用于计算双边谱
#     fft_amp0[0]=0.5*fft_amp0[0]
#     N_2 = int(N/2)
#     fft_amp1 = fft_amp0[0:N_2]  # 单边谱
#     # 计算频谱的频率轴
#     list0 = np.array(range(0, N))
#     list1 = np.array(range(0, int(N/2)))
#     list0_shift = np.array(range(0, N))
#     dt=9.6*1e-8
#     sample_freq=1/dt
#     freq0 = sample_freq*list0/N        # 双边谱的频率轴
#     freq1 = sample_freq*list1/N        # 单边谱的频率轴
#     # # 单边谱
#     max=np.max(fft_amp1[100:])
#     min=np.min(fft_amp1[100:])
#     fft_data_new=fft_data_new+fft_amp1
# fft_amp1=fft_data_new
# plt.figure()
# plt.plot(freq1[10:10006], fft_amp1[10:10006],label='real_data')
# plt.title(' spectrum single-sided')
# plt.legend()
# plt.xlabel('frequency  (Hz)')
# plt.ylabel(' Amplitude ')
# plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/test10/spectrum_real_data.png")

## spectrum_real_data_filter
fft_data_new=np.zeros((5003))
for i in range(27):
    N=10006
    fft_data = fft(real_data_filter[:10006,i])
    fft_amp0 = np.array(np.abs(fft_data)/N*2)   # 用于计算双边谱
    fft_amp0[0]=0.5*fft_amp0[0]
    N_2 = int(N/2)
    fft_amp1 = fft_amp0[0:N_2]  # 单边谱
    # 计算频谱的频率轴
    list0 = np.array(range(0, N))
    list1 = np.array(range(0, int(N/2)))
    list0_shift = np.array(range(0, N))
    dt=9.6*1e-8
    sample_freq=1/dt
    freq0 = sample_freq*list0/N        # 双边谱的频率轴
    freq1 = sample_freq*list1/N        # 单边谱的频率轴
    # # 单边谱
    fft_data_new=fft_data_new+fft_amp1
fft_amp1=fft_data_new
plt.figure()
max=np.max(fft_amp1[10:10006])
plt.plot(freq1[10:1000], fft_amp1[10:1000]/max,label='real_data_filter',linestyle='-.')
plt.title(' spectrum single-sided')
plt.legend()
plt.xlabel('frequency  (Hz)')
plt.ylabel(' Amplitude ')
plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/test10/spectrum_real_data_filter.png")