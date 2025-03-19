import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
# k=25242
k=201
# k=1
j=80
data_real=sio.loadmat('/home/pengyaoguang/data/well_data/{}_{}v_real_test.mat'.format(k,j))['data']
data_updete=sio.loadmat('/home/pengyaoguang/data/well_data/{}_{}v_updete_test.mat'.format(k,j))['data']
data_sm=sio.loadmat('/home/pengyaoguang/data/well_data/{}_{}v_smooth_test.mat'.format(k,j))['data']
data_FWI=sio.loadmat('/home/pengyaoguang/data/well_data/{}_{}v_FWI.mat'.format(k,j))['data']
# res=[50]
plt.figure()
plt.imshow(data_FWI.T,cmap='jet')
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/{}_{}v_FWI.eps".format(k,j),dpi=300)
plt.close()

# plot vertical velocity-depth profile
num_cols = 3
num_rows = 1
hgrid = 100
fig = plt.figure(figsize=(8, 8))
p1=5
p2=50
p3=92
for i in range(num_cols*num_rows):
    ax1 = fig.add_subplot(num_rows,num_cols,i+1)
    if i==0:
        line1, = ax1.plot(data_sm[p1]/1000,np.arange(0,hgrid),color='green',lw=1.5,ls='--',label='Initial')
        line2, = ax1.plot(data_FWI[p1]/1000,np.arange(0,hgrid),color='purple',lw=1.5,ls='--',label='FWI')
        line4, = ax1.plot(data_updete[p1]/1000,np.arange(0,hgrid),color='blue',lw=1.5,ls='--',label='U-net')
        line6, = ax1.plot(data_real[p1]/1000,np.arange(0,hgrid),color='red',lw=1.5,ls='--',label='True')
        
        plt.legend(loc='lower left',bbox_to_anchor=(0, -0.14),
                   borderaxespad=0,ncol=4,columnspacing=5,handletextpad=0.2,
                   edgecolor='black',fontsize=12.5)
    
        plt.title('x = 0.15 km',fontsize=16)
        ax1.invert_yaxis()
        
        ax1.set_ylabel('Depth (km)',fontsize=14)
        plt.yticks(np.arange(0,hgrid,step=20),['0','0.6','1.2','1.8','2.4'],fontsize=14)
        plt.xticks(np.arange(1.0,8,step=1.5),fontsize=14)


    if i==1:
        line1, = ax1.plot(data_sm[p2]/1000,np.arange(0,hgrid),color='green',lw=1.5,ls='--',label='Initial')
        line2, = ax1.plot(data_FWI[p2]/1000,np.arange(0,hgrid),color='purple',lw=1.5,ls='--',label='FWI')
        line4, = ax1.plot(data_updete[p2]/1000,np.arange(0,hgrid),color='blue',lw=1.5,ls='-.',label='U-net')
        line6, = ax1.plot(data_real[p2]/1000,np.arange(0,hgrid),color='red',lw=1.5,ls='--',label='True')

        
        ax1.invert_yaxis()
        plt.title('x = 1.5 km',fontsize=16)
        ax1.set_xlabel('Velocity (km/s)',fontsize=14)
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.xticks(np.arange(1.0,8,step=1.5),fontsize=14)
        
    if i==2:
        line1, = ax1.plot(data_sm[p3]/1000,np.arange(0,hgrid),color='green',lw=1.5,ls='--',label='Initial')
        line2, = ax1.plot(data_FWI[p3]/1000,np.arange(0,hgrid),color='purple',lw=1.5,ls='--',label='FWI')
        line4, = ax1.plot(data_updete[p3]/1000,np.arange(0,hgrid),color='blue',lw=1.5,ls='--',label='U-net')
        line6, = ax1.plot(data_real[p3]/1000,np.arange(0,hgrid),color='red',lw=1.5,ls='--',label='True')

        ax1.invert_yaxis()
        plt.title('x = 2.76 km',fontsize=16)
        
        plt.setp(ax1.get_yticklabels(), visible=False)
        # plt.xticks(np.arange(21.0,6.1,step=1.5),fontsize=14)
        plt.xticks(np.arange(1.0,8,step=1.5),fontsize=14)
plt.tight_layout(pad=0.1, h_pad=1, w_pad=0.8, rect=[0, 0, 0.99, 0.99])

foo_fig = plt.gcf()

foo_fig.savefig('/home/pengyaoguang/data/well_data/well.png')
foo_fig.savefig("/home/pengyaoguang/data/2D_data/2D_test_result/{}_{}well.eps".format(k,j),dpi=300)
# plt.show()
# for i in res:
#     plt.figure
#     plt.subplot(221)
#     plt.plot(range(len(data_real[i])),data_real[i])
#     plt.subplot(222)
#     plt.plot(range(len(data_real[i])),data_real[i])
#     plt.subplot(223)
#     plt.plot(range(len(data_real[i])),data_real[i])
#     plt.savefig('/home/pengyaoguang/data/well_data/well.png')

    
