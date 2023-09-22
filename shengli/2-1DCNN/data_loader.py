import torch
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import RandomSampler
import torchvision
from scipy.ndimage import gaussian_filter

def data_normalization(data):
    
    x_std, x_mean = torch.std_mean(data)
    return (data-x_mean) / x_std

def data_min_max(data):
    
    x_min = torch.min(data)
    x_max = torch.max(data)
    
    return (data - x_min) / (x_max - x_min)

def data_RandomSampler(sample_length, data_x_length, data_y_length, randam_seed = 1234):
    
    # sample_length : number of data you want to sample
    # data_x_length : sampling data from data_x and data_x_length is the length of data_x
    # data_y_length : sampling data from data_y and data_y_length is the length of data_y
    # randam_seed   : fix the random seed so you could obtain the same sample point every time 
    
    # PyTorch random number generator
    torch.manual_seed(randam_seed)
    
    # create an array to save the sampling results
    location_x = np.zeros(sample_length)
    count = 0
    sampler_x = RandomSampler(range(data_x_length))
    for index_x in sampler_x:
        if count == sample_length:
            break
        location_x[count] = index_x
        count += 1
    
    # create an array to save the sampling results
    location_y = np.zeros(sample_length)
    count = 0
    sampler_y = RandomSampler(range(data_y_length))
    for index_y in sampler_y:
        if count == sample_length:
            break
        location_y[count] = index_y
        count += 1
    
    return location_x, location_y

def v_smooth(velocity, sigma=5):
    
    return (torch.tensor(1/gaussian_filter(1/velocity.numpy(), sigma)))

def DataLoad_Train(sample_length, v_smooth_sigma, data_norm = 0, r_seed=1234):
    
    
    # data_norm : using normalization for data, 0 for std and mean, 1 for min max.
    
    
    # Set default dtype to float32
    torch.set_default_dtype(torch.float32)

    # PyTorch random number generator
    torch.manual_seed(1234)

    # Random number generators in other libraries
    np.random.seed(1234)


    velocity_filename = '/home/pengyaoguang/work_space/shengli/data/salt_velocity.mat'
    seismic_data_filename = '/home/pengyaoguang/work_space/shengli/data/salt_seismic_data.mat'
    # results_path = '../results/'

    # velocity data
    data = spio.loadmat(velocity_filename)
    velocity = torch.from_numpy(data[str('velocity')])
    print('velocity.shape', velocity.shape)

    x_length, y_length, z_length = velocity.shape

    # shot_index = 50
    # vmin, vmax = torch.quantile(velocity[:,:,shot_index],
    #                             torch.tensor([0.01, 0.99]))
    # plt.imshow(velocity[1,:,:].T.cpu(), aspect='auto',
    #              cmap='jet')
    # plt.xlabel("Receivers")
    # plt.ylabel("Time Sample")
    # plt.title("Velocity 57")
    # plt.colorbar()

    # plt.tight_layout()
    # plt.savefig(results_path+'velocity.jpg')

    # seismic data
    data = spio.loadmat(seismic_data_filename)
    # print('data.keys()', data.keys())
    seismic_data = torch.from_numpy(data[str('seismic_data')])
    print('seismic_data.shape', seismic_data.shape)

    # shot_index = 50
    # vmin, vmax = torch.quantile(seismic_data[:,:,shot_index],
    #                             torch.tensor([0.01, 0.99]))
    # plt.imshow(seismic_data[1,:,:].cpu().T, aspect='auto',
    #              cmap=plt.cm.seismic, vmin=-vmax, vmax=vmax)
    # plt.xlabel("Receivers")
    # plt.ylabel("Time Sample")
    # plt.title("Seismic Data 57")
    # plt.colorbar()

    # plt.tight_layout()
    # plt.savefig(results_path+'seismic_data.jpg')

    # spio.savemat(save_path + 'Overthrust3D_seismic_data.mat',{'seismic_data':seismic_data.cpu().data.numpy()})
    
    if data_norm == 0:
        seismic_data_nor = data_normalization(seismic_data)
        
    else:
        seismic_data_nor = data_min_max(seismic_data)
#         velocity_nor = velocity / 1000.0
#         velocity_nor_smooth = v_smooth(velocity_nor, v_smooth_sigma)
#         velocity_nor_smooth = data_min_max(velocity_nor_smooth)
    
    velocity_nor = velocity / 1000.0
    velocity_nor_smooth = v_smooth(velocity_nor, v_smooth_sigma)
    velocity_nor_smooth = data_normalization(velocity_nor_smooth)
    
    location_x, location_y = data_RandomSampler(sample_length, x_length, y_length, r_seed)

    train_data = seismic_data_nor[location_x, location_y, :]
    train_data_v_smooth = velocity_nor_smooth[location_x, location_y, :]
    label_data = velocity_nor[location_x, location_y, :]

    print('train_data.shape', train_data.shape)
    print('train_data_v_smooth.shape', train_data_v_smooth.shape)
    print('label_data.shape', label_data.shape)
    print('test_data.shape', seismic_data_nor.shape)
    print('test_label.shape', velocity_nor.shape)

    return train_data, train_data_v_smooth, label_data, seismic_data_nor, velocity_nor_smooth, velocity_nor
