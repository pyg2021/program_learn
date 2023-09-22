from data_loader import DataLoad_Train
from Model_kernel import ImpedanceModel
import torch
import matplotlib.pyplot as plt
import scipy.io as spio
device = torch.device('cuda:1')
torch.set_default_dtype(torch.float)

sample_length = 40
v_smooth_sigma = 5
results_path = '/home/pengyaoguang/work_space/shengli/result_dcnn/'
train_data, train_data_v_smooth, train_label, test_data, v_smooth, test_label = DataLoad_Train(sample_length, v_smooth_sigma)
model_size = train_data.size()
model_val_size = test_data.size()
in_channels = 1
out_channels = 1
is_batchnorm = False
net = ImpedanceModel(in_channels, out_channels, is_batchnorm)
net.to(device)
net.load_state_dict(torch.load('/home/pengyaoguang/work_space/shengli/result_dcnn/net_weight.pkl'))
with torch.no_grad():
        net.eval()
        n=35
        outputs_val = net(test_data.reshape(model_val_size[0]*model_val_size[1], 1, model_val_size[2]).to(device))
        outputs_val = outputs_val.reshape(model_val_size[0], model_val_size[1], model_val_size[2])
        vmin, vmax = torch.quantile(outputs_val[n,:,:].cpu(), torch.tensor([0.01, 0.99]))
        
        plt.figure()
        plt.imshow(outputs_val[n,:,:].T.cpu(), aspect='auto', cmap='jet', vmax=vmax, vmin=vmin)
        plt.xlabel("Receivers")
        plt.ylabel("Time Sample")
        plt.title("Velocity 57")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(results_path  + 'test.jpg')
vmin, vmax = torch.quantile(test_data[n,:,:].cpu(), torch.tensor([0.01, 0.99]))
plt.figure()
plt.imshow(test_data[n,:,:].T.cpu(), aspect='auto', cmap='jet')
plt.xlabel("Receivers")
plt.ylabel("Time Sample")
plt.title("Velocity 57")
plt.colorbar()
plt.tight_layout()
plt.savefig(results_path+"input_real.jpg")

vmin, vmax = torch.quantile(test_label[n,:,:].cpu(), torch.tensor([0.01, 0.99]))
plt.figure()
plt.imshow(test_label[n,:,:].T.cpu(), aspect='auto', cmap='jet')
plt.xlabel("Receivers")
plt.ylabel("Time Sample")
plt.title("Velocity 57")
plt.colorbar()
plt.tight_layout()
plt.savefig(results_path+"real.jpg")