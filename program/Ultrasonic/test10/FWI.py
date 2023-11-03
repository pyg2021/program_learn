import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
import numpy as np
import math
import scipy.io as sio
from scipy.interpolate import interp1d
import os
import time 
start = time.time()
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')
c=1
# ny = 600
# nx = 550
dx = 0.001*c
# v = np.fromfile('/home/pengyaoguang/data/Ultrasonic_data/data/v_model4.bin').reshape(ny, nx)
v = np.fromfile('/home/pengyaoguang/data/Ultrasonic_data/data/v_model4.bin').reshape(6000, -1)[::10,::10]
v_real=torch.from_numpy(v).float()
# v=gaussian_filter(v, 5)
plt.figure()
plt.imshow(v.T)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/FWI_test/v_init0.png")

plt.figure()
plt.imshow(v_real.T)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/FWI_test/v_real.png")
v_real=v_real.to(device)
v_init = torch.tensor(v).float().to(device)
v = v_init.clone()
# v[:,150:]=1845
v.requires_grad_()

n_shots = 26
n_epochs = 1000

n_sources_per_shot = 1
d_source = 10
first_source = 160
source_depth = 5

n_receivers_per_shot = 27
d_receiver = 10  
first_receiver = 70+160 
receiver_depth = 5

freq = 200*1000/c
nt = 2500/c*math.pow(c,0.5)
dt = 2.4*math.pow(10,-8)*c*math.pow(c,0.5)*10
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
    (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
    .repeat(n_shots, n_sources_per_shot, 1).to(device)
)


p=20
# change
wave_real=sio.loadmat("/home/pengyaoguang/data/Ultrasonic_data/data/wave_500_aq1.mat")
x_real=np.squeeze(wave_real["stime"])/1000
y_real=np.squeeze(wave_real["seis"])/1000
y_real[600:]=0
y_real[:300]=0
N=nt
f=interp1d(x_real,y_real,kind='linear')
t=np.arange(0,N)*dt
y_real_0=f(t)
#高通滤波
from scipy.signal import butter, lfilter
from scipy.io import wavfile
b, a = butter(4, 0.1, btype="lowpass")
y_real_0_filter = y_real_0
b, a = butter(4, 0.01, btype="highpass")
y_real_0_filter = lfilter(b, a, y_real_0)
b, a = butter(4, 0.05, btype="lowpass")
y_real_0_filter = lfilter(b, a, y_real_0_filter)


source_amplitudes=torch.Tensor(y_real_0_filter).repeat(n_shots, n_sources_per_shot, 1)
source_amplitudes=source_amplitudes[:,:,150:].to(device)
# out = scalar(
#             v_real, dx, dt,
#             source_amplitudes=source_amplitudes[:,:,:],#150
#             source_locations=source_locations,
#             receiver_locations=receiver_locations,
#             pml_width=[p,p,0,p],
#             pml_freq=freq,
#         )
ob_data=sio.loadmat("/home/pengyaoguang/data/Ultrasonic_data/data/real_data_all_test.mat")["data"]
ob_data=torch.from_numpy(ob_data).float()
observed_data =ob_data[:,:,:].to(device)
# observed_data=out[-1]
# for i in range(observed_data.shape[0]):
#     observed_data[i]=(observed_data[i]-torch.mean(observed_data[i]))/math.sqrt(torch.var(observed_data[i]))
# Setup optimiser to perform inversion
optimiser = torch.optim.SGD([v], lr=0.1, momentum=0.9)
loss_fn = torch.nn.MSELoss()

# Run optimisation/inversion
# v_true = v_true.to(device)
def closure():
        optimiser.zero_grad()
        out = scalar(
            v, dx, dt,
            source_amplitudes=source_amplitudes[i:i+m,:,:],#150
            source_locations=source_locations[i:i+m,:,:],
            receiver_locations=receiver_locations[i:i+m,:,:],
            pml_width=[p,p,0,p],
            pml_freq=freq,
        )
        for j in range(out[-1][:,i:,:].shape[1]):
             a=28
             b=440
             out[-1][:,i:,:][:,j,:a*j+b]=0
        # plt.figure(figsize=(20,20))
        # plt.imshow(out[-1][:,i:,::2].cpu().T.detach(),aspect='auto',cmap='seismic')
        # plt.colorbar()
        # plt.xlabel("Channel")
        # plt.ylabel("Time Sample")
        # plt.savefig("/home/pengyaoguang/data/Ultrasonic_data/test10/synthetic_data{}.png".format(i))
        
        loss = 1e5*loss_fn((out[-1][:,i:,::2]-torch.mean(out[-1][:,i:,::2]))/math.sqrt(torch.var(out[-1][:,i:,::2])), observed_data[i:i+m,i:,:])+10e5*loss_fn(torch.clamp(v,1500,1845),v)
        loss.backward()
        torch.nn.utils.clip_grad_value_(
            v,
            torch.quantile(v.grad.detach().abs(), 0.98)
        )
        if i==0:
            print("loss:",loss.cpu().item())
        return loss
m=1
for epoch in range(n_epochs):
    for i in range(0,n_shots,m):
        optimiser.step(closure)
        # with torch.no_grad():
        #      v=torch.maximum(v,torch.ones_like(v)*1800)
    if epoch%50==0:
        plt.figure()
        plt.imshow((v.cpu().detach().numpy().T))
        plt.colorbar()
        plt.savefig("/home/pengyaoguang/data/FWI_test/{}.png".format(epoch))
        end=time.time()
        print("epoch:",epoch,"shot:",i,"time:",end-start,"s")
        plt.close()

sio.savemat("/home/pengyaoguang/data/FWI_test/v.mat",{"data":v.cpu().detach().numpy()})



# plt.figure()

# plt.imshow(observed_data[1].T,vmax=0.1,vmin=-0.1,cmap='seismic')
# plt.colorbar()
# plt.savefig('000.png')
# plt.figure()

# plt.imshow(observed_data[1].T,vmax=0.1,vmin=-0.1,aspect='auto',cmap='seismic')
# plt.colorbar()
# plt.savefig('000.png')
