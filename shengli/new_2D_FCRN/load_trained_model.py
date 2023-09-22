from data_loader import DataLoad_Train
from Model_FCRN import Model_FCRN
import torch
import matplotlib.pyplot as plt
import scipy.io as spio

in_channels = 1
out_channels = 1
is_batchnorm = False
net = Model_FCRN(in_channels, out_channels, is_batchnorm)
device = torch.device('cuda')
net.to(device)
net.load_state_dict(torch.load('shengli/results/net_weight.pkl'))
sample_length = 2
v_smooth_sigma = 5
data_norm = 0
r_seed = 123
train_seismic_data, train_data_v_smooth, train_label, test_seismic_data, v_smooth, test_label = DataLoad_Train(sample_length, 
                                                                                                               v_smooth_sigma, 
                                                                                                               data_norm, r_seed)
model_val_size = test_seismic_data.size()
test_data = test_seismic_data.reshape(model_val_size[0]*model_val_size[1], 1, model_val_size[2])
with torch.no_grad():
    n=50
    net.eval()
    outputs_val = net(test_data.to(device))
    outputs_val = outputs_val.reshape(model_val_size[0], model_val_size[1], model_val_size[2])
    vmin, vmax = torch.quantile(outputs_val[n,:,:].cpu(), torch.tensor([0.01, 0.99]))

    plt.figure()
    plt.imshow(outputs_val[n,:,:].T.cpu(), aspect='auto', cmap='jet', vmax=vmax, vmin=vmin)
    plt.xlabel("Receivers")
    plt.ylabel("Time Sample")
    plt.title("Velocity 57")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("shengli/results/trained_test.jpg")
vmin, vmax = torch.quantile(test_seismic_data[n,:,:].cpu(), torch.tensor([0.01, 0.99]))
plt.figure()
plt.imshow(test_seismic_data[n,:,:].T.cpu(), aspect='auto', cmap='jet')
plt.xlabel("Receivers")
plt.ylabel("Time Sample")
plt.title("Velocity 57")
plt.colorbar()
plt.tight_layout()
plt.savefig("shengli/results/input_real.jpg")

vmin, vmax = torch.quantile(test_label[n,:,:].cpu(), torch.tensor([0.01, 0.99]))
plt.figure()
plt.imshow(test_label[n,:,:].T.cpu(), aspect='auto', cmap='jet')
plt.xlabel("Receivers")
plt.ylabel("Time Sample")
plt.title("Velocity 57")
plt.colorbar()
plt.tight_layout()
plt.savefig("shengli/results/real.jpg")