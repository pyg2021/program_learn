import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ny = 2301
nx = 751
dx = 4.0
v_true = torch.from_file('./deepwave_learn/marmousi_vp.bin',
                         size=ny*nx).reshape(ny, nx)

# Select portion of model for inversion
ny = 600
nx = 250
v_true = v_true[:ny, :nx]

# Smooth to use as starting model
v_init = torch.tensor(1/gaussian_filter(1/v_true.numpy(), 40)).to(device)
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
    torch.from_file('./deepwave_learn/marmousi_data.bin',
                    size=n_shots*n_receivers_per_shot*nt)
    .reshape(n_shots, n_receivers_per_shot, nt)
)

# Select portion of data for inversion
n_shots = 20
n_receivers_per_shot = 100
nt = 300
observed_data = observed_data[:n_shots, :n_receivers_per_shot, :nt].to(device)

# Scale observed data to have unit RMS amplitude
observed_data /= (observed_data**2).mean().sqrt().item()

# source_locations
source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                               dtype=torch.long, device=device)
source_locations[..., 1] = source_depth
source_locations[:, 0, 0] = torch.arange(n_shots) * d_source + first_source

# receiver_locations
receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                 dtype=torch.long, device=device)
receiver_locations[..., 1] = receiver_depth
receiver_locations[:, :, 0] = (
    (torch.arange(n_receivers_per_shot) * d_receiver + first_receiver)
    .repeat(n_shots, 1)
)

# source_amplitudes
source_amplitudes = (
    (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
    .repeat(n_shots, n_sources_per_shot, 1).to(device)
)

# Setup optimiser to perform inversion
optimiser = torch.optim.LBFGS([v])
loss_fn = torch.nn.MSELoss()

# Run optimisation/inversion
n_epochs = 5
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
        predicted = out[-1]
        predicted_norm = (predicted.detach()**2).mean().sqrt().item()
        if predicted_norm > 0:
            normed_predicted = predicted / predicted_norm
        else:
            normed_predicted = predicted
        lower_mask = torch.nn.functional.relu(-(v.detach()-1200))
        upper_mask = torch.nn.functional.relu(v.detach()-6000)
        loss = (1e6*loss_fn(normed_predicted, observed_data) +
                (((v-1200)**2)*lower_mask).mean() +
                (((v-6000)**2)*upper_mask).mean())
        loss.backward()
        return loss

    optimiser.step(closure)

# Plot
vmin = v_true.min()
vmax = v_true.max()
_, ax = plt.subplots(3, figsize=(10.5, 10.5), sharex=True, sharey=True)
ax[0].imshow(v_init.cpu().T, aspect='auto', cmap='gray',
             vmin=vmin, vmax=vmax)
ax[0].set_title("Initial")
ax[1].imshow(v.detach().cpu().T, aspect='auto', cmap='gray',
             vmin=vmin, vmax=vmax)
ax[1].set_title("Out")
ax[2].imshow(v_true.cpu().T, aspect='auto', cmap='gray',
             vmin=vmin, vmax=vmax)
ax[2].set_title("True")
plt.tight_layout()
plt.savefig('./deepwave_learn/example_normalised_fwi.jpg')

v.detach().cpu().numpy().tofile('./deepwave_learn/marmousi_v_inv.bin')