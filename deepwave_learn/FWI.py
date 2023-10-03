import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')
ny = 2301
nx = 751
dx = 4.0
v_true = torch.from_file('/home/pengyaoguang/data/vp.bin',
                         size=ny*nx).reshape(ny, nx)

# Select portion of model for inversion
ny = 600
nx = 250
v_true = v_true[:ny, :nx]


v_init = (torch.tensor(1/gaussian_filter(1/v_true.numpy(), 40))
          .to(device))
v = v_init.clone()
v.requires_grad_()

n_shots = 115

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

observed_data = (
    torch.from_file('/home/pengyaoguang/data/marmousi_data.bin',
                    size=n_shots*n_receivers_per_shot*nt)
    .reshape(n_shots, n_receivers_per_shot, nt)
)

n_shots = 20
n_receivers_per_shot = 100
nt = 300
observed_data = (
    observed_data[:n_shots, :n_receivers_per_shot, :nt].to(device)
)

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

# Setup optimiser to perform inversion
optimiser = torch.optim.SGD([v], lr=0.1, momentum=0.9)
loss_fn = torch.nn.MSELoss()

# Run optimisation/inversion
n_epochs = 25
v_true = v_true.to(device)

for epoch in range(n_epochs):
    def closure():
        optimiser.zero_grad()
        out = scalar(
            v, dx, dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_freq=freq,
        )
        loss = 1e10 * loss_fn(out[-1], observed_data)
        loss.backward()
        torch.nn.utils.clip_grad_value_(
            v,
            torch.quantile(v.grad.detach().abs(), 0.98)
        )
        print("epoch",epoch,"loss:",loss.cpu().item())
        return loss

    optimiser.step(closure)
    plt.imshow((v.cpu().detach().numpy().T))
    plt.savefig("/home/pengyaoguang/data/FWI_test/FWI_normal/{}.png".format(epoch))




# plt.figure()

# plt.imshow(observed_data[1].T,vmax=0.1,vmin=-0.1,cmap='seismic')
# plt.colorbar()
# plt.savefig('000.png')
# plt.figure()

# plt.imshow(observed_data[1].T,vmax=0.1,vmin=-0.1,aspect='auto',cmap='seismic')
# plt.colorbar()
# plt.savefig('000.png')
