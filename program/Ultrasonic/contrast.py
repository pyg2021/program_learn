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

# original = real_data
# dft = cv2.dft(np.float32(original), flags=cv2.DFT_COMPLEX_OUTPUT)   # 傅里叶变换
# fshift = np.fft.fftshift(dft)    # 低频移至中心
# # 定义高通滤波器
# # 设置掩膜
# rows, cols = original.shape
# crow, ccol = int(rows / 2), int(cols / 2)   # 中心位置
# mask = np.ones((rows, cols, 2), np.uint8)
# mask[crow-1:crow+1, ccol-1:ccol+1] = 0
# f = fshift * mask     # 将掩模与傅里叶变化后的图像相乘，保留四周部分，即保留高频部分
# ishift = np.fft.ifftshift(f)       # 低频移回
# img_back = cv2.idft(ishift)     # 傅里叶逆变换
# img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])   # 频域转回空域
# real_data_filter=img_back

synthetic=sio.loadmat('/home/pengyaoguang/data/Ultrasonic_data/synthetic_data/synthetic_data0.mat')
synthetic_data=synthetic['V'][0].T

plt.figure(figsize=(20,20))
# plt.subplot(121)
# u=np.mean(real_data)
# var=np.var(real_data)
# real_data=(real_data-u)/var
# vmax=np.max(real_data[40:,:1])
# n=1
# plt.figure()
# plt.imshow(real_data/vmax,aspect='auto',cmap='seismic', vmin=-n, vmax=n,extent=(1,27,0.960576,0))
# plt.colorbar()
# plt.subplot(122)

# plt.imshow(real_data[:5208,:1]/vmax,aspect='auto',cmap='seismic', vmin=-n/2, vmax=n/2,extent=(0,26,0.5,0))
# t=np.arange(40,5208)*9.6e-5
# plt.plot(t,real_data[40:5208,:1],label='real_data')
# vamx=np.max(real_data_filter[40:,:1])
# plt.plot(t,real_data_filter[40:5208,:1]/vmax,label='filter_real_data')
# plt.legend()
# plt.savefig("program/Ultrasonic/result/real_data2.png")
# plt.figure(figsize=(20,20))
# vmax=np.max(synthetic_data[:20833,])
# m=0.005
# t=np.arange(0,20833)*2.4e-5
# plt.plot(t,synthetic_data[:20833,:1]/vmax,label='synthetic_data')
# plt.legend()
# plt.xlabel("ms")
plt.figure()
max=np.max(synthetic_data[:25000,])
plt.imshow(synthetic_data[:25000,]/max,aspect='auto',cmap='seismic',vmax=0.005,vmin=-0.005,extent=(0,26,0.564,0))
plt.colorbar()
# plt.colorbar()
# plt.xlabel("Channel")
# plt.ylabel("Time Sample(ms)")
# plt.set_cmap("gray")
plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/data/synthetic_data.png")
plt.figure()
max=np.max(real_data[:6250,:])
plt.imshow(real_data[:6250,:]/max,aspect='auto',cmap='seismic',vmax=1,vmin=-1,extent=(0,26,0.6,0))
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/data/real_data.png")
plt.figure()
max=np.max(real_data_filter[:6250,:])
plt.imshow(real_data_filter[100:6250,:]/max,aspect='auto',cmap='seismic',vmax=0.03,vmin=-0.03,extent=(0,26,0.6,0))
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/data/real_data_filter.png")
shot_1_filter=sio.loadmat('/home/pengyaoguang/1325/program/Ultrasonic/shot_1_filter.mat')
shot_1_filter=shot_1_filter['sht_a_f']
plt.figure()
max=np.max(shot_1_filter[:6250,:])
plt.imshow(shot_1_filter[:6250,:]/max,aspect='auto',cmap='seismic',vmax=1,vmin=-1,extent=(0,26,0.6,0))
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/data/shot_1_filter.png")

fft_data_new=np.zeros((11750))
from scipy.fftpack import fft,fftshift
for i in range(27):
    N=23500
    fft_data = fft(synthetic_data[:23500,i])
    fft_amp0 = np.array(np.abs(fft_data)/N*2)   # 用于计算双边谱
    fft_amp0[0]=0.5*fft_amp0[0]
    N_2 = int(N/2)
    fft_amp1 = fft_amp0[0:N_2]  # 单边谱
    # 计算频谱的频率轴
    list0 = np.array(range(0, N))
    list1 = np.array(range(0, int(N/2)))
    list0_shift = np.array(range(0, N))
    dt=2.4*1e-8
    sample_freq=1/dt
    freq0 = sample_freq*list0/N        # 双边谱的频率轴
    freq1 = sample_freq*list1/N        # 单边谱的频率轴
    # # 单边谱
    max=np.max(fft_amp1[3:])
    min=np.min(fft_amp1[3:])
    fft_data_new=fft_data_new+fft_amp1
fft_amp1=fft_data_new
# plt.subplot(222)
plt.figure()
# fft_data_new=np.zeros((12500))
# for i in range(fft_data.shape[1]):
#     fft_data_new=fft_data_new[:]+fft_amp1[:,i]
# fft_amp1=fft_data_new
nmax=np.max(fft_amp1[:25000])
plt.plot(freq1[:5000], fft_amp1[:5000]/nmax,label='synthetic_data')
plt.legend()
plt.title(' spectrum single-sided')
# plt.ylim(0, 0.01)
plt.xlabel('frequency  (Hz)')
plt.ylabel(' Amplitude ')
plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/data/synthetic_data_frequency.png")

water=sio.loadmat('/home/pengyaoguang/data/Ultrasonic_data/data/air_injection_scan_pre_switch.mat')
real_data=water['all_traces_pre_switch_CSG'][0].item().T
fft_data_new=np.zeros((5003))
for i in range(27):
    N=10006
    fft_data = fft(real_data[:10006,i])
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
    max=np.max(fft_amp1[100:])
    min=np.min(fft_amp1[100:])
    fft_data_new=fft_data_new+fft_amp1
fft_amp1=fft_data_new
# plt.subplot(222)
# plt.figure()
# plt.plot(freq1[10:10006], fft_amp1[10:10006],label='real_data')
# plt.title(' spectrum single-sided')
# plt.legend()
# # plt.ylim(0, 0.01)
# plt.xlabel('frequency  (Hz)')
# plt.ylabel(' Amplitude ')
# # plt.legend()
# plt.savefig("m1.png")

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
    max=np.max(fft_amp1[100:])
    min=np.min(fft_amp1[100:])
    fft_data_new=fft_data_new+fft_amp1
fft_amp1=fft_data_new
# plt.subplot(222)
# plt.figure()
max=np.max(fft_amp1[10:10006])
plt.plot(freq1[10:10006], fft_amp1[10:10006]/max,label='filter_data')
plt.title(' spectrum single-sided')
plt.legend()
# plt.ylim(0, 0.01)
plt.xlabel('frequency  (Hz)')
plt.ylabel(' Amplitude ')
# plt.legend()
plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/data/real_data_filter_frequency.png")