import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
import os
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
device = torch.device('cuda:1' if torch.cuda.is_available()
                      else 'cpu')
ny = 2301
nx = 751
dx = 4.0
v = torch.from_file('/mnt/sda/home/yaoguang/deepwave_space/vp.bin',
                    size=ny*nx).reshape(ny, nx).to(device)
n_shots_all=115
n_shots = 4
time_start_1 = time.time()


n_sources_per_shot = 1
d_source = 20  # 20 * 4m = 80m
first_source = 10  # 10 * 4m = 40m
source_depth = 2  # 2 * 4m = 8m

n_receivers_per_shot = 384
d_receiver = 6  # 6 * 4m = 24m
first_receiver = 0  # 0 * 4m = 0m
receiver_depth = 2  # 2 * 4m = 8m

freq = 25
nt = 750
dt = 0.004
peak_time = 1.5 / freq

receiver_amplitudes_all=torch.ones(n_shots_all,n_receivers_per_shot,nt)
for number in range(n_shots,n_shots_all+n_shots,n_shots):
    # source_locations
    source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                dtype=torch.long, device=device)
    source_locations[..., 1] = source_depth
    source_locations[:, 0, 0] = (torch.arange(number-n_shots,number) * d_source +
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
                pml_freq=freq)
    # print(out)

    receiver_amplitudes = out[-1]
    receiver_amplitudes_all[number-n_shots:number,:,:]=receiver_amplitudes[:,:,:]


time_end_1 = time.time()
print("time:"+str(time_end_1 - time_start_1)+"s")

#figure
vmin, vmax = torch.quantile(receiver_amplitudes_all[0],
                            torch.tensor([0.05, 0.95]).to(device))
f, ax = plt.subplots(1, 2, figsize=(10.5, 7), sharey=True)
print(ax[0])
ax[0].imshow(receiver_amplitudes_all[59].cpu().T, aspect='auto',
                cmap='gray', vmin=vmin, vmax=vmax)
ax[1].imshow(receiver_amplitudes_all[:, 192].cpu().T, aspect='auto',
                cmap='gray', vmin=vmin, vmax=vmax)
ax[0].set_xlabel("Channel")
ax[0].set_ylabel("Time Sample")
ax[1].set_xlabel("Shot")
plt.tight_layout()

f.savefig("./test/whole_figure_test.png")

receiver_amplitudes_all.cpu().numpy().tofile('./test/marmousi_data_test.bin')