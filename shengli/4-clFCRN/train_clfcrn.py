from data_loader_clfcrn import DataLoad_Train
from Model_clFCRN import Network_clFCRN
import torch
import matplotlib.pyplot as plt
import scipy.io as spio
import time
start=time.time()

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
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

# Setup optimiser to perform inversion
optimizer = torch.optim.Adam(net.parameters())
loss_fn = torch.nn.L1Loss()
# loss_fn_2 = torch.nn.MSELoss()

num_epochs = 500000
for epoch in range(num_epochs):
    
    optimizer.zero_grad()
    n=100000
    recon_seismic_data, pre_velocity, recon_seismic_data_pre_v, pre_seismic_data, recon_velocity = net(seismic_data[:n].to(device), 
                                                                             pre_seismic_data_label.to(device), 
                                                                             pre_velocity_label.to(device))
    
    loss = (
            loss_fn(recon_seismic_data, seismic_data[:n].to(device)) + 
        
            loss_fn(pre_velocity, pre_velocity_label.to(device)) + 
            loss_fn(recon_seismic_data_pre_v, pre_seismic_data_label.to(device)) + 
        
            loss_fn(pre_seismic_data, pre_seismic_data_label.to(device)) + 
            loss_fn(recon_velocity, pre_velocity_label.to(device))
        )
    loss.backward()
    
    optimizer.step()
    
    if epoch % 50000 == 0:
        with torch.no_grad():
            print(epoch,loss.item())
            net.eval()
            
            outputs_val = net.encode(seismic_data.to(device))
            outputs_val = outputs_val.reshape(seismic_data_size[0], seismic_data_size[1], seismic_data_size[2])
            vmin, vmax = torch.quantile(outputs_val[0,:,:].cpu(), torch.tensor([0.01, 0.99]))
            
            plt.figure()
            plt.imshow(outputs_val[0,:,:].T.cpu(), aspect='auto', cmap='jet', vmax=vmax, vmin=vmin)
            plt.xlabel("Receivers")
            plt.ylabel("Time Sample")
            plt.title("Velocity 57")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(results_path + str(epoch) + '_pre_velocity_sm.jpg')
torch.save(net.state_dict(),results_path+'net_weight.pkl')
vmin, vmax = torch.quantile(velocity_nor[0,:,:].cpu(), torch.tensor([0.01, 0.99]))
plt.figure()
plt.imshow(velocity_nor[0,:,:].T.cpu(), aspect='auto', cmap='jet', vmax=vmax, vmin=vmin)
plt.xlabel("Receivers")
plt.ylabel("Time Sample")
plt.title("Velocity 57")
plt.colorbar()
plt.tight_layout()
plt.savefig(results_path+'real_v.jpg')

vmin, vmax = torch.quantile(seismic_data_nor[0,:,:].cpu(), torch.tensor([0.01, 0.99]))
plt.figure()
plt.imshow(seismic_data_nor[0,:,:].T.cpu(), aspect='auto', cmap='jet', vmax=vmax, vmin=vmin)
plt.xlabel("Receivers")
plt.ylabel("Time Sample")
plt.title("Velocity 57")
plt.colorbar()
plt.tight_layout()
plt.savefig(results_path+'real_seismic_data.jpg')
end=time.time()
print(end-start,"s")