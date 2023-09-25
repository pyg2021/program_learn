import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
import scipy.io as sio
import numpy as np
import time
start=time.time()
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')
ny = 301
nx = 201
dx = 10
i=1
for i in range(1,1601):
    v=sio.loadmat("/home/pengyaoguang/data/unet_data/train_data/SimulateData/vmodel_train/vmodel{}.mat".format(i))['vmodel']
    # ny = 2301
    # nx = 751
    # dx = 4.0
    # v = torch.from_file('/home/pengyaoguang/data/vp.bin',
    #                     size=ny*nx).reshape(ny, nx)
    # plt.figure()
    # plt.imshow(v)
    # plt.colorbar()
    # plt.savefig("/home/pengyaoguang/data/unet_data/result/svmodel1.png")
    v=np.float32(v.T)
    v=torch.from_numpy(v).to(device)
    n_shots = 27

    n_sources_per_shot = 1
    d_source = 10  # 20 * 4m = 80m
    first_source = 10  # 10 * 4m = 40m
    source_depth = 0  # 2 * 4m = 8m

    n_receivers_per_shot = 301
    d_receiver = 1  # 6 * 4m = 24m
    first_receiver = 0  # 0 * 4m = 0m
    receiver_depth = 0  # 2 * 4m = 8m

    freq = 20
    nt = 2000
    dt = 0.001
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

    out = scalar(v, dx, dt, source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                accuracy=8,
                pml_freq=freq)

    receiver_amplitudes = out[-1]
    # vmin, vmax = torch.quantile(receiver_amplitudes[0],
    #                             torch.tensor([0.01, 0.99]).to(device))
    # _, ax = plt.subplots(1, 2, figsize=(10.5, 7), sharey=True)
    # ax[0].imshow(receiver_amplitudes[0].cpu().T, aspect='auto',
    #             cmap='seismic', vmin=-vmax, vmax=vmax)
    # ax[1].imshow(receiver_amplitudes[:, 10].cpu().T, aspect='auto',
    #             cmap='seismic', vmin=vmin, vmax=vmax)
    # ax[0].set_xlabel("Channel")
    # ax[0].set_ylabel("Time Sample")
    # ax[1].set_xlabel("Shot")
    # plt.tight_layout()


    # plt.savefig('/home/pengyaoguang/data/unet_data/result/test_image.png')
    end=time.time()
    print(end-start,"ss")

    seismic=receiver_amplitudes.cpu().numpy()
    sio.savemat("/home/pengyaoguang/data/unet_data/train_data/SimulateData/georec_train/seismic_data{}.mat".format(i),{'seismic_data':seismic})
    # receiver_amplitudes.cpu().numpy().tofile('./deepwave_learn/marmousi_data.bin')
