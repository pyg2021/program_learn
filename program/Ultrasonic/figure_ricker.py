import math
import deepwave
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

freq = 900*1000
nt = 25000
dt = 2.4*math.pow(10,-8)
peak_time = 1.5 / freq

m=deepwave.wavelets.ricker(freq, nt, dt, peak_time)
length=nt
t = torch.arange(float(length)) * dt - peak_time
y = (1 - 2 * math.pi**2 * freq**2 * t**2) \
        * torch.exp(-math.pi**2 * freq**2 * t**2)
plt.figure()
n=4000
plt.plot(t[:n],y[:n])
print(m.shape)
plt.show()
plt.savefig('ricker1.png')

n=1000
wave_real=sio.loadmat("program/Ultrasonic/data/wave_500_aq1.mat")
x_real=np.squeeze(wave_real["stime"])
y_real=np.squeeze(wave_real["seis"])
plt.plot(x_real[:n]/1000,y_real[:n]/1000)


plt.legend(["ricker", "wave_real"]) 
plt.xlabel("ms")
plt.savefig('wave2.png')
