import deepwave
import torch
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt

# Set default dtype to float32
# torch.set_default_dtype(torch.float32)
# PyTorch random number generator

def deconv_data(m):
    torch.manual_seed(1234)
    # Random number generators in other libraries
    np.random.seed(1234)
    device = torch.device('cpu')

    velocity_filename = '/home/pengyaoguang/data/3D_v_model/v{}.mat'.format(m)
    # velocity_filename = '/home/pengyaoguang/data/Overthrust3D.mat'
    save_path = '/home/pengyaoguang/data/3D_net_result/useful_result/'
    data = spio.loadmat(velocity_filename)
    torch.set_default_dtype(torch.float32)

    velocity = torch.from_numpy(data[str('v')]).float()
    # velocity = torch.from_numpy(data[str('velocity')]).float()
    # velocity = torch.from_numpy(data[str('velocity')])
    # print(velocity.shape)
    shot_index = 50
    vmin, vmax = torch.quantile(velocity[:,:,shot_index],
                                torch.tensor([0.01, 0.99]))
    # plt.imshow(velocity[50,:,:].T.cpu(), aspect='auto',
    #              cmap='jet')
    # plt.xlabel("Receivers")
    # plt.ylabel("Time Sample")
    # plt.title("Velocity 57")
    # plt.colorbar()
    # # plt.tight_layout()
    # # plt.savefig(result_path+'observed data.jpg')
    # plt.savefig(save_path + 'salt_velocity.png')
    # spio.savemat(save_path + 'salt_velocity.mat',{'velocity':velocity.cpu().data.numpy()})

    reflectivity = torch.zeros_like(velocity)
    n=velocity.shape[2]
    reflectivity[:,:,0:n-1] = (velocity[:,:,1:n] - velocity[:,:,0:n-1]) / (velocity[:,:,1:n] + velocity[:,:,0:n-1])
    shot_index = 50
    vmin, vmax = torch.quantile(reflectivity[:,:,shot_index],
                                torch.tensor([0.01, 0.99]))
    # plt.figure()
    # plt.imshow(reflectivity[50,:,:].T.cpu(), aspect='auto',
    #              cmap=plt.cm.seismic, vmin=-vmax, vmax=vmax)
    # plt.xlabel("Receivers")
    # plt.ylabel("Time Sample")
    # plt.title("Reflectivity 57")
    # plt.colorbar()
    # plt.tight_layout()
    # # plt.savefig(result_path+'observed data.jpg')
    # plt.savefig(save_path + 'salt_reflectivity.png')
    # spio.savemat(save_path + 'salt_reflectivity.mat',{'reflectivity':reflectivity.cpu().data.numpy()})

    freq = 10
    nt = 100
    dt = 0.01
    peak_time = 1.0 / freq

    # source_amplitudes
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time).to(device)
    )

    source_amplitudes_inverse = torch.flip(source_amplitudes.cpu(),[0])

    # print(source_amplitudes.shape)
    # plt.figure()
    # plt.plot(source_amplitudes.cpu())
    # plt.plot(source_amplitudes_inverse)
    # plt.show()
    # plt.savefig(save_path + 'salt_amplitudes.png')
    source_amplitudes_matrix = torch.zeros(nt,nt)
    for index_i in range(nt):
        source_amplitudes_matrix[index_i,0:index_i+1] =  source_amplitudes_inverse[nt-index_i-1:nt]
        
    #     plt.plot(source_amplitudes_matrix[index_i,:])
    #     plt.show()



    seismic_data = torch.matmul(reflectivity,source_amplitudes_matrix.T)
    # print(torch.matmul(reflectivity,source_amplitudes_matrix.T).shape)

    shot_index = 50
    vmin, vmax = torch.quantile(seismic_data[:,:,shot_index],
                                torch.tensor([0.01, 0.99]))
    # plt.figure()
    # plt.imshow(seismic_data[50,:,:].cpu().T, aspect='auto',
    #              cmap=plt.cm.seismic, vmin=-vmax, vmax=vmax)
    # plt.xlabel("Receivers")
    # plt.ylabel("Time Sample")
    # plt.title("Seismic Data 57")
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig(save_path + 'salt_seismic.png')
    # # plt.savefig(result_path+'observed data.jpg')
    # spio.savemat(save_path + 'salt_seismic_data.mat',{'seismic_data':seismic_data.cpu().data.numpy()})
    return seismic_data
deconv_data(29998)