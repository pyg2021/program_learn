import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar,scalar_born
import math
from scipy.ndimage import gaussian_filter
import time
import scipy.io as sio

start=time.time()
device = torch.device('cuda:1' if torch.cuda.is_available()
                      else 'cpu')
ny = 100
nx = 100
dx = 10.0
for k in range(25000,25001):
    v_0=sio.loadmat("/home/pengyaoguang/data/3D_v_model/v{}".format(k))["v"]*1000
    for j in range(v_0.shape[0]):
        v=v_0[j]
        plt.imshow(v.T)
        plt.savefig('/home/pengyaoguang/data/2D_data/2D_v_model/v{}_{}.jpg'.format(k,j))
        
        # Smooth to use as migration model
        v_mig = torch.tensor(1/gaussian_filter(1/v, 40))
        v_mig = v_mig.float().to(device)
        ny = v_mig.shape[0]
        nx = v_mig.shape[1]
        v=torch.from_numpy(v).float().to(device)
        v.detach().cpu().numpy().tofile('/home/pengyaoguang/data/2D_data/2D_v_model/v{}_{}.bin'.format(k,j))
        n_shots = 45

        n_sources_per_shot = 1
        d_source = 2  
        first_source = 0  
        source_depth = 1

        n_receivers_per_shot = 90
        d_receiver = 1  
        first_receiver = 0 
        receiver_depth = 1 

        freq = 15
        nt = 200
        dt = 0.005
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

        # Propagate
        out = scalar(v, dx, dt, source_amplitudes=source_amplitudes,
                    source_locations=source_locations,
                    receiver_locations=receiver_locations,
                    accuracy=8,
                    pml_freq=freq)

        # Plot
        receiver_amplitudes = out[-1]
        vmin, vmax = torch.quantile(receiver_amplitudes[0],
                                    torch.tensor([0.05, 0.95]).to(device))
        _, ax = plt.subplots(1, 2, figsize=(10.5, 7), sharey=True)
        ax[0].imshow(receiver_amplitudes[22].cpu().T, aspect='auto',
                    cmap='gray', vmin=vmin, vmax=vmax)
        ax[1].imshow(receiver_amplitudes[:, 45].cpu().T, aspect='auto',
                    cmap='gray', vmin=vmin, vmax=vmax)
        ax[0].set_xlabel("Channel")
        ax[0].set_ylabel("Time Sample")
        ax[1].set_xlabel("Shot")
        plt.tight_layout()
        plt.savefig('/home/pengyaoguang/data/2D_data/2D_RTM/forward_model{}_{}.jpg'.format(k,j))


        # Load observed data



        observed_data=receiver_amplitudes
        # Create mask to attenuate direct arrival
        mask = torch.ones_like(observed_data)
        flat_len = 40
        taper_len = 80
        taper = torch.cos(torch.arange(taper_len)/taper_len * math.pi/2)
        mute_len = flat_len + 2*taper_len
        mute = torch.zeros(mute_len, device=device)
        mute[:taper_len] = taper
        mute[-taper_len:] = taper.flip(0)
        v_direct = torch.min(v)
        for shot_idx in range(n_shots):
            sx = (shot_idx * d_source + first_source) * dx
            for receiver_idx in range(n_receivers_per_shot):
                rx = (receiver_idx * d_receiver + first_receiver) * dx
                dist = abs(sx - rx)
                arrival_time = dist / v_direct / dt
                mute_start = int(arrival_time) - mute_len//2
                mute_end = mute_start + mute_len
                if (mute_start > nt):
                    continue
                actual_mute_start = max(mute_start, 0)
                actual_mute_end = min(mute_end, nt)
                mask[shot_idx, receiver_idx,
                    actual_mute_start:actual_mute_end] = \
                    mute[actual_mute_start-mute_start:
                        actual_mute_end-mute_start]
        observed_scatter_masked = observed_data * mask

        vmin, vmax = torch.quantile(observed_data[0],
                                    torch.tensor([0.05, 0.95]).to(device))
        _, ax = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True,
                            sharey=True)
        ax[0].imshow(observed_data[0].cpu().T, aspect='auto', cmap='gray',
                    vmin=vmin, vmax=vmax)
        ax[0].set_title("Observed")
        ax[1].imshow(mask[0].cpu().T, aspect='auto', cmap='gray')
        ax[1].set_title("Mask")
        ax[2].imshow(observed_scatter_masked[0].cpu().T, aspect='auto',
                    cmap='gray', vmin=vmin, vmax=vmax)
        ax[2].set_title("Masked data")
        plt.tight_layout()
        plt.savefig('/home/pengyaoguang/data/2D_data/2D_RTM/rtm_mask{}_{}.jpg'.format(k,j))




        # Create scattering amplitude that we will invert for
        scatter = torch.zeros_like(v_mig)
        scatter.requires_grad_()

        # Setup optimiser to perform inversion
        optimiser = torch.optim.SGD([scatter], lr=1e9)
        loss_fn = torch.nn.MSELoss()

        # Run optimisation/inversion
        n_epochs = 1
        n_batch = 46  # The number of batches to use
        n_shots_per_batch = (n_shots + n_batch - 1) // n_batch
        for epoch in range(n_epochs):
            epoch_loss = 0
            optimiser.zero_grad()
            for batch in range(n_batch):
                batch_start = batch * n_shots_per_batch
                batch_end = min(batch_start + n_shots_per_batch,
                                n_shots)
                if batch_end <= batch_start:
                    continue
                s = slice(batch_start, batch_end)
                out = scalar_born(
                    v_mig, scatter, dx, dt,
                    source_amplitudes=source_amplitudes[s],
                    source_locations=source_locations[s],
                    receiver_locations=receiver_locations[s],
                    pml_freq=freq
                )
                loss = (
                    loss_fn(out[-1] * mask[s],
                            observed_scatter_masked[s])
                )
                epoch_loss += loss.item()
                loss.backward()
            print(epoch_loss)
            optimiser.step()

        # Plot
        vmin, vmax = torch.quantile(scatter.detach(),
                                    torch.tensor([0.05, 0.95]).to(device))
        plt.figure(figsize=(10.5, 3.5))
        plt.imshow(scatter.detach().cpu().T, aspect='auto', cmap='gray',
                vmin=vmin, vmax=vmax)
        plt.savefig('/home/pengyaoguang/data/2D_data/2D_RTM/rtm{}_{}.jpg'.format(k,j))

        scatter.detach().cpu().numpy().tofile('/home/pengyaoguang/data/2D_data/2D_RTM/RTM{}_{}.bin'.format(k,j))
        end=time.time()
        print('finish:',end-start,'s')
        # break