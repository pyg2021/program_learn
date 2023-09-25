import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
import os
import numpy as np
import math
import scipy.io as sio
import time
from scipy.interpolate import interp1d
time_start_1 = time.time()
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0' # 下面老是报错 shape 不一致
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')
# device=torch.device("cpu")
c=1
ny = 6000
nx = 5500
dx = 0.0001*c
v = np.fromfile('/home/pengyaoguang/data/v_model4.bin').reshape(ny, nx)
# v=np.random.uniform(1.12,1.15,(2301,751))
v=torch.from_numpy(v).float()
v=v.to(device)
n_shots = 1
for i in range(1):
    n_sources_per_shot = 1
    d_source = 100
    first_source = 1600
    source_depth = 50  

    n_receivers_per_shot = 27
    d_receiver = 100  
    first_receiver = 1600+700 
    receiver_depth = 50

    freq = 900*1000/c
    nt = 25000/c*math.pow(c,0.5)
    dt = 2.4*math.pow(10,-8)*c*math.pow(c,0.5)
    print(dt)
    peak_time = 1.5 / freq

    # source_locations
    source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                dtype=torch.long, device=device)
    source_locations[..., 1] = source_depth
    source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source +
                                first_source)

    # receiver_locations
    receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                    dtype=torch.long, device=device)
    receiver_locations[..., 1] = receiver_depth
    receiver_locations[:, :, 0] = (
        (torch.arange(n_receivers_per_shot) * d_receiver +
        first_receiver)
        .repeat(n_shots, 1)
    )

    # source_amplitudes
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
        .repeat(n_shots, n_sources_per_shot, 1)
        .to(device)
    )
    p=20

    # change
    wave_real=sio.loadmat("program/Ultrasonic/data/wave_500_aq1.mat")
    x_real=np.squeeze(wave_real["stime"])/1000
    y_real=np.squeeze(wave_real["seis"])/1000
    y_real[600:]=0
    N=25000
    f=interp1d(x_real,y_real,kind='linear')
    t=np.arange(0,25000)*2.4*1e-8
    y_real_0=f(t)
    # y_real[0:10]=0
    # y_real[700:]=0
    plt.figure()
    plt.plot(t[:4000],y_real_0[:4000],label='inter')
    # plt.plot(x_real,y_real,label='real')
    plt.legend()
    plt.savefig("t0.png")
    #高通滤波
    from scipy.signal import butter, lfilter
    from scipy.io import wavfile
    b, a = butter(4, 0.01, btype="highpass")
    y_real_0_filter = lfilter(b, a, y_real_0)
    plt.figure()
    plt.plot(t,y_real_0_filter,label='filter')
    plt.legend()
    plt.savefig("t.png")
    # y_real[0:300]=0
    # y_real[600:]=0
    from scipy.fftpack import fft,fftshift
    # N=25000
    # f=interp1d(x_real,y_real,kind='linear')
    # t=np.arange(0,25000)*2.4*1e-8
    # y_real=f(t)
    # plt.figure()
    # plt.plot(t[1000:5000],y_real_0[1000:5000],label='real_wave')
    # plt.plot(t[1000:5000],y_real[1000:5000],label='filter_wave')
    # plt.legend()
    # plt.savefig("t3.png")


    fft_data = fft(y_real_0_filter)
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
    max=max(fft_amp1[:])
    min=min(fft_amp1[:])
    # plt.subplot(222)
    plt.figure()
    plt.plot(freq1[:], fft_amp1[:],label='filter')
    plt.title(' spectrum single-sided')
    # plt.ylim(0, 0.01)
    plt.xlabel('frequency  (Hz)')
    plt.ylabel(' Amplitude ')
    


    fft_data = fft(y_real_0)
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
    max=np.max(fft_amp1[:])
    min=np.min(fft_amp1[:])
    # plt.subplot(222)
    plt.plot(freq1[:], fft_amp1[:],'--',label='real')
    plt.legend()
    plt.title(' spectrum single-sided')
    # plt.ylim(0, 0.01)
    plt.xlabel('frequency  (Hz)')
    plt.ylabel(' Amplitude ')
    plt.savefig("m.png")
    source_amplitudes=y_real_0_filter[np.newaxis,np.newaxis,:]
    source_amplitudes=torch.Tensor(source_amplitudes).to(device)
    





    # nt = 25000
    # dt = 2.4*math.pow(10,-8)
    # peak_time = 1.5 / freq


    # length=nt
    # t = np.arange(float(length)) * dt - peak_time
    # y = (1 - 2 * math.pi**2 * freq**2 * t**2) \
    #         * np.exp(-math.pi**2 * freq**2 * t**2)
    # y=y[np.newaxis,np.newaxis,:]
    # source_amplitudes=torch.Tensor(y).to(device)


    # 8th order accurate spatial finite differences
    out = scalar(v, dx, dt, source_amplitudes=source_amplitudes[:,:,1500:],
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                accuracy=8,
                pml_width=[p,p,0,p],
                pml_freq=freq)

    time_end_1 = time.time()
    print("time:"+str(time_end_1 - time_start_1)+"s")


    receiver_amplitudes = out[-1]
    print(torch.max(receiver_amplitudes))
    vmin, vmax = torch.quantile(receiver_amplitudes[0],
                                torch.tensor([0.01, 0.99]).to(device))
    # vmin=torch.min(receiver_amplitudes[0]).cpu()
    # vmax=torch.max(receiver_amplitudes[0]).cpu()
    # f, ax = plt.subplots(1, 2, figsize=(20, 20), sharey=True)
    # print(ax[0])
    # ax[0].imshow(receiver_amplitudes[0].cpu().T, aspect='auto',
    #              cmap='seismic', vmin=-vmax*150, vmax=vmax*150)
    # ax[1].imshow(receiver_amplitudes[:, 20].cpu().T, aspect='auto',
    #              cmap='seismic', vmin=-vmax*150, vmax=vmax*150)
    # ax[0].set_xlabel("Channel")
    # ax[0].set_ylabel("Time Sample")
    # ax[1].set_xlabel("Shot")
    # plt.tight_layout()
    # plt.colorbar
    # f.savefig("program/Ultrasonic/whole_figurebig2.png")

    # plt.figure(1)
    plt.figure(figsize=(20,20))
    plt.imshow(receiver_amplitudes[0][:,:].cpu().T,aspect='auto',cmap='seismic', vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.xlabel("Channel")
    plt.ylabel("Time Sample")
    # plt.set_cmap("gray")
    # plt.savefig("program/Ultrasonic/result/a{}.png".format(i))
    plt.savefig("program/Ultrasonic/result/a4.png")

# plt.imsave("program/Ultrasonic/whole_figure2.png",receiver_amplitudes[10,:,:].cpu().detach().T)
from scipy.io import savemat
savemat('program/Ultrasonic/data4.mat',{'V':receiver_amplitudes.cpu().numpy()})
# receiver_amplitudes.cpu().numpy().tofile('program/Ultrasonic/data2.bin')