import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
import numpy as np
import scipy.io as sio
import time
import os
start=time.time()
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
device = torch.device('cuda:1')
ny = 100
nx = 100
dx = 4
# k=25242
k=1
# k=201
# j=50
j=50
all_data=torch.ones((100,100,100))
for j in range(100):
    print(k,j)
    v_true = torch.Tensor(sio.loadmat("/home/pengyaoguang/data/3D_RTM2/v{}".format(k-1))["v"][j]*1000).float()
    # v_true = torch.Tensor(sio.loadmat("/home/pengyaoguang/data/3D_v_model/v{}".format(k))["v"][j]*1000).float()

    # Select portion of model for inversion
    # ny = 600
    # nx = 250
    # v_true = v_true[:ny, :nx]


    v_init = (torch.tensor(1/gaussian_filter(1/v_true.numpy(), 40))
            .to(device))
    # plt.figure()
    # plt.imshow((v_init.cpu().detach().numpy().T))
    # plt.savefig("/home/pengyaoguang/program_learn/2D/2d_Rtm_data/fwi/start_5.png")
    # plt.close()
    v = v_init.clone()
    v.requires_grad_()

    n_shots = 49

    n_sources_per_shot = 1
    d_source = 2  # 20 * 4m = 80m
    first_source = 0  # 10 * 4m = 40m
    source_depth = 1  # 2 * 4m = 8m

    n_receivers_per_shot = 99
    d_receiver =1   # 6 * 4m = 24m
    first_receiver = 0  # 0 * 4m = 0m
    receiver_depth = 1  # 2 * 4m = 8m

    freq = 15
    nt = 750
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
    v_true = v_true.to(device)
    # plt.figure()
    # plt.imshow((v_true.cpu().detach().numpy().T))
    # plt.colorbar()
    # plt.savefig("/home/pengyaoguang/program_learn/2D/2d_Rtm_data/fwi/true_5.png")
    # plt.close()
    out = scalar(v_true, dx, dt, source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                accuracy=8,
                pml_freq=freq)

    receiver_amplitudes = out[-1]


    # plt.figure()
    # vmax,vmin=torch.quantile(receiver_amplitudes[50].cpu(),torch.tensor([0.02,0.98]))
    # vmax=torch.max(receiver_amplitudes[50].cpu())
    # vmin=torch.min(receiver_amplitudes[50].cpu())
    # plt.imshow(receiver_amplitudes[50].detach().cpu().T,aspect='auto',cmap="gray",vmax=vmax,vmin=vmin)
    # plt.colorbar()
    # plt.savefig("/home/pengyaoguang/program_learn/2D/2d_Rtm_data/0.png")
    observed_data = receiver_amplitudes


    # Setup optimiser to perform inversion
    optimiser = torch.optim.SGD([v], lr=0.1, momentum=0.9)
    # optimiser = torch.optim.AdamW([v],lr=1e-3)
    # optimiser = torch.optim.AdamW([v],lr=1e-3)
    # optimiser = torch.optim.LBFGS([v],
    #                     history_size=10,
    #                     max_iter=4,
    #                     line_search_fn="strong_wolfe")
    loss_fn = torch.nn.MSELoss()
    loss_1=torch.nn.L1Loss()
    from skimage.metrics import structural_similarity as ssim
    import math
    def ssim_metric(target: object, prediction: object, win_size: int=21):
        cur_ssim = ssim(
            target,
            prediction,
            win_size=win_size,
            data_range=target.max() - target.min(),
        )
        return cur_ssim
    # Run optimisation/inversion
    def SNR_singlech(S, SN):
        PS = torch.mean(torch.square(S))
        PN = torch.mean(torch.square(SN-S))
        # PS = torch.sum((S - mean_S) ** 2) # 纯信号的功率
        # PN = torch.sum((S - SN) ** 2) # 噪声的功率
        snr = 10 * math.log10(PS / PN) # 计算信噪比
        return snr
    n_epochs = 400

    res=[]
    def closure():
        optimiser.zero_grad()
        out = scalar(
            v, dx, dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_freq=freq,
        )
        # loss = 1e9 * loss_fn(out[-1], observed_data)+0.01*loss_fn(v[::10],v_true[::10])+0.01*total_variation_loss(v)
        loss = 1e9 * loss_fn(out[-1], observed_data)
        # print(1e9 * loss_fn(out[-1], observed_data),0.1*loss_fn(v[::10],v_true[::10]),0.1*total_variation_loss(v))
        loss.backward()
        if epoch%10==0:
            print("epoch",epoch,"loss:",loss.cpu().item())
        return loss

    def total_variation_loss(image, weight=1.0):
        # 获取图像的形状
        height, width = image.size()
        
        # 计算水平方向的梯度
        horizontal_diff = image[ :, 1:] - image[ :, :-1]
        # 计算垂直方向的梯度
        vertical_diff = image[1:, :] - image[:-1, :]
        
        # 计算梯度的绝对值
        abs_horizontal_diff = torch.abs(horizontal_diff)
        abs_vertical_diff = torch.abs(vertical_diff)
        
        # 计算TV正则化项（L1范数）
        tv_loss = weight * (torch.sum(abs_horizontal_diff) + torch.sum(abs_vertical_diff))
        
        # 注意：分母中的减项是为了去除边缘像素，因为这些像素在某一方向上没有相邻像素
        # 如果你希望包括边缘像素，可以调整分母的计算方式
        
        return tv_loss
    number=0
    sign=1
    for epoch in range(n_epochs):
        # if epoch>100:
        number+=1
        if number>100:
            print(sign)
            if sign==1:
                print('-----------------------------------------------')
                optimiser = torch.optim.LBFGS([v],
                            history_size=10,
                            max_iter=4,
                            line_search_fn="strong_wolfe")
            else:
                print('???????????????????????????????????????????????')
                optimiser = torch.optim.SGD([v], lr=0.1, momentum=0.9)
            sign=-sign

            number=0
        optimiser.step(closure)
        torch.nn.utils.clip_grad_value_(
            v,
            torch.quantile(v.grad.detach().abs(), 0.98)
        )
        
        # res.append(loss.cpu().item())
        if epoch%50==0:
            print((time.time()-start)/60,'min')
            # plt.figure()
            # plt.imshow((v.cpu().detach().numpy().T))
            # plt.colorbar()
            # plt.savefig("/home/pengyaoguang/program_learn/2D/2d_Rtm_data/fwi/update_5.png".format(epoch))
            # # plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/{}_{}v_FWI.eps".format(k,j),dpi=300)
            # plt.close()
            # sio.savemat('/home/pengyaoguang/data/well_data/{}_{}v_FWI.mat'.format(k,j),{'data':v.cpu().detach()})
            print('L2:',loss_fn(v,v_true),"L1",loss_1(v,v_true),'ssim:',ssim_metric(v.cpu().detach().numpy(),v_true.cpu().detach().numpy()),'SNR:',SNR_singlech(v_true,v))
            # plt.figure()
            # plt.plot(range(len(res)),res)
            # plt.savefig("/home/pengyaoguang/program_learn/2D/2d_Rtm_data/fwi/hi_2.png".format(epoch))
            # plt.close()
        # break
    # break
    all_data[j]=v.cpu().detach()
    # break
sio.savemat('/home/pengyaoguang/data/3D_fuse/{}_v_FWI.mat'.format(k),{'v':all_data/1000})


# plt.figure()

# plt.imshow(observed_data[1].T,vmax=0.1,vmin=-0.1,cmap='seismic')
# plt.colorbar()
# plt.savefig('000.png')
# plt.figure()

# plt.imshow(observed_data[1].T,vmax=0.1,vmin=-0.1,aspect='auto',cmap='seismic')
# plt.colorbar()
# plt.savefig('000.png')
