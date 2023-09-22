from data_loader_clfcrn import DataLoad_Train
from Model_clFCRN import Network_clFCRN
import torch
import matplotlib.pyplot as plt
import scipy.io as spio

device = torch.device('cuda:1')
torch.set_default_dtype(torch.float)

sample_length = 2
v_smooth_sigma = 5
data_norm = 1
r_seed = 123
results_path = '/home/pengyaoguang/work_space/shengli/results_clfcrn/'

seismic_data_nor, velocity_nor, velocity_nor_smooth, location_x, location_y = DataLoad_Train(sample_length, v_smooth_sigma,data_norm, r_seed)
pre_seismic_data_label = seismic_data_nor[location_x, location_y, :]
pre_velocity_label = velocity_nor[location_x, location_y, :]

pre_data_size = pre_seismic_data_label.size()
pre_seismic_data_label = pre_seismic_data_label.unsqueeze(1)
pre_velocity_label = pre_velocity_label.unsqueeze(1)

seismic_data_size = seismic_data_nor.size()
seismic_data = seismic_data_nor.reshape(seismic_data_size[0]*seismic_data_size[1], seismic_data_size[2]).unsqueeze(1)

# pre_seismic_data_label.shape, pre_velocity_label.shape, seismic_data.shape

in_channels = 1
out_channels = 1
is_batchnorm = False
net = Network_clFCRN(in_channels, out_channels, is_batchnorm)
net.to(device)

net.load_state_dict(torch.load('shengli/results_clfcrn/net_weight.pkl'))

with torch.no_grad():
    net.eval()
    n=50
    outputs_val = net.encode(seismic_data.to(device))
    outputs_val = outputs_val.reshape(seismic_data_size[0], seismic_data_size[1], seismic_data_size[2])
    vmin, vmax = torch.quantile(outputs_val[n,:,:].cpu(), torch.tensor([0.01, 0.99]))

    plt.figure()
    plt.imshow(outputs_val[n,:,:].T.cpu(), aspect='auto', cmap='jet', vmax=vmax, vmin=vmin)
    plt.xlabel("Receivers")
    plt.ylabel("Time Sample")
    plt.title("Velocity 57")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(results_path + 'test.jpg')

vmin, vmax = torch.quantile(velocity_nor[n,:,:].cpu(), torch.tensor([0.01, 0.99]))
plt.figure()
plt.imshow(velocity_nor[n,:,:].T.cpu(), aspect='auto', cmap='jet', vmax=vmax, vmin=vmin)
plt.xlabel("Receivers")
plt.ylabel("Time Sample")
plt.title("Velocity 57")
plt.colorbar()
plt.tight_layout()
plt.savefig(results_path+'real_v.jpg')

vmin, vmax = torch.quantile(seismic_data_nor[n,:,:].cpu(), torch.tensor([0.01, 0.99]))
plt.figure()
plt.imshow(seismic_data_nor[n,:,:].T.cpu(), aspect='auto', cmap='jet', vmax=vmax, vmin=vmin)
plt.xlabel("Receivers")
plt.ylabel("Time Sample")
plt.title("Velocity 57")
plt.colorbar()
plt.tight_layout()
plt.savefig(results_path+'real_seismic_data.jpg')
