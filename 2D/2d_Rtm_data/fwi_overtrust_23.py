import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
import numpy as np
import scipy.io as sio
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')
ny = 100
nx = 100
dx = 4.0
k=0
v_true = torch.tensor(sio.loadmat("/home/pengyaoguang/data/3D_RTM2/v{}".format(k))["v"][50]*1000).float()

# Select portion of model for inversion
# ny = 600
# nx = 250
# v_true = v_true[:ny, :nx]


v_init = (torch.tensor(1/gaussian_filter(1/v_true.numpy(), 40))
          .to(device))
v = v_init.clone()
v.requires_grad_()

n_shots = 99

n_sources_per_shot = 1
d_source = 1  # 20 * 4m = 80m
first_source = 0  # 10 * 4m = 40m
source_depth = 2  # 2 * 4m = 8m

n_receivers_per_shot = 99
d_receiver =1   # 6 * 4m = 24m
first_receiver = 0  # 0 * 4m = 0m
receiver_depth = 2  # 2 * 4m = 8m

freq = 15
nt = 750
dt = 0.0004
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
v_true = v_true.to(device)
plt.figure()
plt.imshow((v_true.cpu().detach().numpy().T))
plt.savefig("/home/pengyaoguang/program_learn/2D/2d_Rtm_data/fwi/2.png")
plt.close()
out = scalar(v_true, dx, dt, source_amplitudes=source_amplitudes,
             source_locations=source_locations,
             receiver_locations=receiver_locations,
             accuracy=8,
             pml_freq=freq)

receiver_amplitudes = out[-1]


plt.figure()
plt.imshow(receiver_amplitudes[0].detach().cpu().T,aspect='auto')
plt.savefig("/home/pengyaoguang/program_learn/2D/2d_Rtm_data/0.png")
observed_data = receiver_amplitudes


# Setup optimiser to perform inversion
optimiser = torch.optim.SGD([v], lr=0.1, momentum=0.9)
loss_fn = torch.nn.MSELoss()

# Run optimisation/inversion
n_epochs = 10000


for epoch in range(n_epochs):
    optimiser.zero_grad()
    out = scalar(
        v, dx, dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_freq=freq,
    )
    loss = 1e8 * loss_fn(out[-1], observed_data)
    # loss =  loss_fn(out[-1], observed_data)
    loss.backward()
    optimiser.step()
    torch.nn.utils.clip_grad_value_(
        v,
        torch.quantile(v.grad.detach().abs(), 0.98)
    )
    print("epoch",epoch,"loss:",loss.cpu().item())
    if epoch%100==0:
        plt.figure()
        plt.imshow((v.cpu().detach().numpy().T))
        plt.savefig("/home/pengyaoguang/program_learn/2D/2d_Rtm_data/fwi/3.png".format(epoch))
        plt.close()




# plt.figure()

# plt.imshow(observed_data[1].T,vmax=0.1,vmin=-0.1,cmap='seismic')
# plt.colorbar()
# plt.savefig('000.png')
# plt.figure()

# plt.imshow(observed_data[1].T,vmax=0.1,vmin=-0.1,aspect='auto',cmap='seismic')
# plt.colorbar()
# plt.savefig('000.png')
