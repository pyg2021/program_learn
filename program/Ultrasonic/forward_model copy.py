import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
import os
import numpy as np
import math
import scipy
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
device = torch.device('cuda:1' if torch.cuda.is_available()
                      else 'cpu')
c=1
ny = 600
nx = 550
dx = 0.0001*c
v = np.fromfile('program/Ultrasonic/v_mode0.bin').reshape(ny, nx)
# v=np.random.uniform(1.12,1.15,(2301,751))
v=torch.from_numpy(v).float()
v=v.to(device)
n_shots = 1

n_sources_per_shot = 1
d_source = 10  
first_source = 181  
source_depth = 5  

n_receivers_per_shot = 400
d_receiver = 1  
first_receiver = 91  
receiver_depth = 5  

freq = 900*1000/c
nt = 3000
dt = 2.5*math.pow(10,-8)*c
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

# 8th order accurate spatial finite differences
out = scalar(v, dx, dt, source_amplitudes=source_amplitudes,
             source_locations=source_locations,
             receiver_locations=receiver_locations,
             accuracy=8,
             pml_width=20,
             pml_freq=freq)
# print(out)

receiver_amplitudes = out[-1]
vmin, vmax = torch.quantile(receiver_amplitudes[0],
                            torch.tensor([0.01, 0.99]).to(device))
f, ax = plt.subplots(1, 2, figsize=(20, 20), sharey=True)
print(ax[0])
ax[0].imshow(receiver_amplitudes[0].cpu().T, aspect='auto',
             cmap='seismic', vmin=vmin, vmax=vmax)
ax[1].imshow(receiver_amplitudes[:, 2].cpu().T, aspect='auto',
             cmap='seismic', vmin=vmin, vmax=vmax)
ax[0].set_xlabel("Channel")
ax[0].set_ylabel("Time Sample")
ax[1].set_xlabel("Shot")
plt.tight_layout()
f.savefig("program/Ultrasonic/whole_figure_samll.png")
# plt.imsave("program/Ultrasonic/whole_figure2.png",receiver_amplitudes[10,:,:].cpu().detach().T)
from scipy.io import savemat
savemat('program/Ultrasonic/data2.mat',{'V':receiver_amplitudes.cpu().numpy()})
receiver_amplitudes.cpu().numpy().tofile('program/Ultrasonic/data2.bin')