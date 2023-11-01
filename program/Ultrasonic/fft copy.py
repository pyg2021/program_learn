import numpy as np
from scipy.fftpack import fft,fftshift
import matplotlib.pyplot as plt
import scipy.io as sio
import math


N = 10000                        # 采样点数
sample_interval=9.6*1e-8   # 采样间隔
sample_freq=1/ sample_interval   # 采样频率 120 Hz, 大于两倍的最高频率
signal_len=N*sample_interval    # 信号长度
t=np.arange(0,signal_len,sample_interval)

wave_real=sio.loadmat("program/Ultrasonic/data/wave_500_aq1.mat")
x_real=np.squeeze(wave_real["stime"])
y_real=np.squeeze(wave_real["seis"])
signal = y_real/1000 # 采集的信号

from scipy.signal import butter, lfilter
from scipy.io import wavfile

b, a = butter(4, 0.01, btype="highpass")
signal = lfilter(b, a, signal)



fft_data = fft(signal)
# 在python的计算方式中，fft结果的直接取模和真实信号的幅值不一样。
# 对于非直流量的频率，直接取模幅值会扩大N/2倍， 所以需要除了N乘以2。
# 对于直流量的频率(0Hz)，直接取模幅值会扩大N倍，所以需要除了N。
fft_amp0 = np.array(np.abs(fft_data)/N*2)   # 用于计算双边谱
fft_amp0[0]=0.5*fft_amp0[0]
N_2 = int(N/2)
fft_amp1 = fft_amp0[0:N_2]  # 单边谱
fft_amp0_shift = fftshift(fft_amp0)    # 使用fftshift将信号的零频移动到中间

# 计算频谱的频率轴
list0 = np.array(range(0, N))
list1 = np.array(range(0, int(N/2)))
list0_shift = np.array(range(0, N))
freq0 = sample_freq*list0/N        # 双边谱的频率轴
freq1 = sample_freq*list1/N        # 单边谱的频率轴
freq0_shift=sample_freq*list0_shift/N-sample_freq/2  # 零频移动后的频率轴

# 绘制结果
plt.figure()
# 原信号
plt.subplot(221)
plt.plot(t[:700], signal[:700])
plt.title(' Original signal')
plt.xlabel('t (s)')
plt.ylabel(' Amplitude ')
# 双边谱
# plt.subplot(222)
# plt.plot(freq0, fft_amp0)
# plt.title(' spectrum two-sided')
# plt.ylim(0, 0.001*60)
# plt.xlabel('frequency  (Hz)')
# plt.ylabel(' Amplitude ')
# # 单边谱
max=max(fft_amp1[3:])
min=min(fft_amp1[3:])
plt.subplot(222)
plt.plot(freq1[0:3000], (fft_amp1[0:3000]-min)/(max-min))
plt.title(' spectrum single-sided')
# plt.ylim(0, 0.01)
plt.xlabel('frequency  (Hz)')
plt.ylabel(' Amplitude ')
# # 移动零频后的双边谱
# plt.subplot(224)
# plt.plot(freq0_shift, fft_amp0_shift)
# plt.title(' spectrum two-sided shifted')
# plt.xlabel('frequency  (Hz)')
# plt.ylabel(' Amplitude ')
# plt.ylim(0, 6)

freq = 900*1000
nt = 37500
dt = 2.4*math.pow(10,-8)
peak_time = 1.5 / freq


length=nt
t = np.arange(float(length)) * dt - peak_time
y = (1 - 2 * math.pi**2 * freq**2 * t**2) \
        * np.exp(-math.pi**2 * freq**2 * t**2)

N = 37500                        # 采样点数
sample_interval=2.4*1e-8   # 采样间隔
sample_freq=1/sample_interval   # 采样频率 120 Hz, 大于两倍的最高频率
signal_len=N*sample_interval    # 信号长度
t=np.arange(0,signal_len,sample_interval)
signal = y # 采集的信号

fft_data = fft(signal)

# 在python的计算方式中，fft结果的直接取模和真实信号的幅值不一样。
# 对于非直流量的频率，直接取模幅值会扩大N/2倍， 所以需要除了N乘以2。
# 对于直流量的频率(0Hz)，直接取模幅值会扩大N倍，所以需要除了N。
fft_amp0 = np.array(np.abs(fft_data)/N*2)   # 用于计算双边谱
fft_amp0[0]=0.5*fft_amp0[0]
N_2 = int(N/2)
fft_amp1 = fft_amp0[0:N_2]  # 单边谱
fft_amp0_shift = fftshift(fft_amp0)    # 使用fftshift将信号的零频移动到中间

# 计算频谱的频率轴
list0 = np.array(range(0, N))
list1 = np.array(range(0, int(N/2)))
list0_shift = np.array(range(0, N))
freq0 = sample_freq*list0/N        # 双边谱的频率轴
freq1 = sample_freq*list1/N        # 单边谱的频率轴
freq0_shift=sample_freq*list0_shift/N-sample_freq/2  # 零频移动后的频率轴

# 原信号
plt.subplot(221)
plt.plot(t[:3000], signal[:3000])
# plt.title(' Original signal')
plt.xlabel('t (s)')
plt.ylabel(' Amplitude ')
# 双边谱
# plt.subplot(224)
# plt.plot(freq0, fft_amp0)
# # plt.title(' spectrum two-sided')
# plt.ylim(0, 6/1000)
# plt.xlabel('frequency  (Hz)')
# plt.ylabel(' Amplitude ')
# # 单边谱
max=np.max(fft_amp1[3:])
min=np.min(fft_amp1[3:])
plt.subplot(222)
plt.plot(freq1[3:3000], (fft_amp1[3:3000]-min)/(max-min))
plt.title(' spectrum single-sided')
# plt.ylim(0, 0.001)
plt.xlabel('frequency  (Hz)')
plt.ylabel(' Amplitude ')

plt.show()
plt.savefig("all3.png")
print()