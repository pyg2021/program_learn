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
v = np.fromfile('program/Ultrasonic/v_model4.bin').reshape(ny, nx)
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

    # 8th order accurate spatial finite differences
    step_ratio=1
    target_abs_max=1
    nt=23500
    # dt, step_ratio = deepwave.common.cfl_condition(dx, dx, dt, 2000)
    for i in range(0,math.floor(nt)):
        chunk = source_amplitudes[..., i*step_ratio:(i+1)*step_ratio]
        if i == 0:
            out = deepwave.scalar(v, dx, dt,
                                source_amplitudes=chunk,
                                source_locations=source_locations,
                                pml_width=[20,20,0,20])
        else:
            out = deepwave.scalar(v, dx, dt,
                                source_amplitudes=chunk,
                                source_locations=source_locations,
                                pml_width=[20,20,0,20],
                                wavefield_0=out[0],
                                wavefield_m1=out[1],
                                psiy_m1=out[2],
                                psix_m1=out[3],
                                zetay_m1=out[4],
                                zetax_m1=out[5])
        val = out[0][0] / target_abs_max


        if i%500==0:
            plt.figure(figsize=(20,20))
            # vmax=torch.max(val)
            plt.imshow(val.cpu().T,aspect='auto',cmap='gray',vmin=-5e-11,vmax=5e-11)
            plt.colorbar()
            plt.xlabel("Channel")
            plt.ylabel("Time Sample")
            # plt.set_cmap("gray")
            plt.savefig("program/Ultrasonic/png3/{}.png".format(i))
            # torchvision.utils.save_image(val, f'deepwave_learn/result2/wavefield_{i:06d}.jpg')
    
    time_end_1 = time.time()
    print("time:"+str(time_end_1 - time_start_1)+"s")