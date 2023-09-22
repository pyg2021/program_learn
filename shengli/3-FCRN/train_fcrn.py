from data_loader import DataLoad_Train
from Model_FCRN import Model_FCRN
import torch
import matplotlib.pyplot as plt
import scipy.io as spio
import time
start=time.time()
# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device('cuda:1')
torch.set_default_dtype(torch.float)

sample_length = 40
v_smooth_sigma = 5
data_norm = 0
r_seed = 123
results_path = 'shengli/results/'
train_seismic_data, train_data_v_smooth, train_label, test_seismic_data, v_smooth, test_label = DataLoad_Train(sample_length, 
                                                                                                               v_smooth_sigma, 
                                                                                                               data_norm, r_seed)

model_size = train_seismic_data.size()
model_val_size = test_seismic_data.size()
train_data = train_seismic_data.view(model_size[0],1,model_size[1])
test_data = test_seismic_data.reshape(model_val_size[0]*model_val_size[1], 1, model_val_size[2])
test_data.shape
in_channels = 1
out_channels = 1
is_batchnorm = False
net = Model_FCRN(in_channels, out_channels, is_batchnorm)
net.to(device)
# Setup optimiser to perform inversion
optimizer = torch.optim.AdamW(net.parameters())
loss_fn_1 = torch.nn.L1Loss()
# loss_fn_2 = torch.nn.MSELoss()

num_epochs = 5000
for epoch in range(num_epochs):
    
    optimizer.zero_grad()
    
    outputs = net(train_data.to(device))
    
#     loss = (
#             loss_fn_1(outputs,
#                     (train_label.view(model_size[0], 1, model_size[1]).to(device))) + 
#             loss_fn_2(outputs,
#                     (train_label.view(model_size[0], 1, model_size[1]).to(device)))
#         )
    loss = (
            loss_fn_1(outputs,
                    (train_label.view(model_size[0], 1, model_size[1]).to(device)))
        )
    loss.backward()
    
    optimizer.step()
    
    if epoch % 50000 == 0:
        with torch.no_grad():
            print(epoch,loss.item())
            net.eval()
            
            outputs_val = net(test_data.to(device))
            outputs_val = outputs_val.reshape(model_val_size[0], model_val_size[1], model_val_size[2])
            vmin, vmax = torch.quantile(outputs_val[0,:,:].cpu(), torch.tensor([0.01, 0.99]))
            
            plt.figure()
            plt.imshow(outputs_val[0,:,:].T.cpu(), aspect='auto', cmap='jet', vmax=vmax, vmin=vmin)
            plt.xlabel("Receivers")
            plt.ylabel("Time Sample")
            plt.title("Velocity 57")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(results_path + str(epoch) + '_pre_velocity_sm.jpg')
# torch.save(net.state_dict(),results_path+'net_weight.pkl')
vmin, vmax = torch.quantile(test_seismic_data[0,:,:].cpu(), torch.tensor([0.01, 0.99]))
plt.figure()
plt.imshow(test_seismic_data[0,:,:].T.cpu(), aspect='auto', cmap='jet')
plt.xlabel("Receivers")
plt.ylabel("Time Sample")
plt.title("Velocity 57")
plt.colorbar()
plt.tight_layout()
plt.savefig(results_path +'input_real.jpg')

vmin, vmax = torch.quantile(test_label[0,:,:].cpu(), torch.tensor([0.01, 0.99]))
plt.figure()
plt.imshow(test_label[0,:,:].T.cpu(), aspect='auto', cmap='jet')
plt.xlabel("Receivers")
plt.ylabel("Time Sample")
plt.title("Velocity 57")
plt.colorbar()
plt.tight_layout()
plt.savefig(results_path +'real.jpg')
end=time.time()
print(end-start,"s")
# 10min 500000