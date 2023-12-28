if __name__ == "__main__":
    USE_GPU_AWARE_DASK = True
    from examples.seismic import demo_model,Model
    import matplotlib.pyplot as plt
    from devito import configuration
    import time
    import scipy.io as sio
    from devito import gaussian_smooth
    from examples.seismic import AcquisitionGeometry
    import numpy as np
    from examples.seismic.acoustic import AcousticWaveSolver
    from distributed import Client, wait
    import gc
    # import sys

    # Serial modeling function
    def forward_modeling_single_shot(model, geometry, save=False, dt=4.0):
        solver = AcousticWaveSolver(model, geometry, space_order=4)
        d_obs, u0 = solver.forward(vp=model.vp, save=save)[0:2]
        # del solver,model,geometry
        for key, value in globals().items():
            if key == "d_obs" or "u0" or "dt":
                continue
            del globals()[key]
        return d_obs.resample(dt), u0


    # Parallel modeling function
    def forward_modeling_multi_shots(model, geometry, save=False, dt=4.0):

        futures = []
        for i in range(geometry.nsrc):

            # Geometry for current shot
            geometry_i = AcquisitionGeometry(model, geometry.rec_positions, geometry.src_positions[i,:], 
                geometry.t0, geometry.tn, f0=geometry.f0, src_type=geometry.src_type)
            
            # Call serial modeling function for each index
            futures.append(client.submit(forward_modeling_single_shot, model, geometry_i, save=save, dt=dt))

        # Wait for all workers to finish and collect shots
        wait(futures)
        shots = []
        for i in range(geometry.nsrc):
            shots.append(futures[i].result()[0])

        return shots
    # Serial FWI objective function
    def fwi_objective_single_shot(model, geometry, d_obs):

        # Devito objects for gradient and data residual
        grad = Function(name="grad", grid=model.grid)
        residual = Receiver(name='rec', grid=model.grid,
                            time_range=geometry.time_axis, 
                            coordinates=geometry.rec_positions)
        solver = AcousticWaveSolver(model, geometry, space_order=4)

        # Predicted data and residual
        d_pred, u0 = solver.forward(vp=model.vp, save=True)[0:2]
        residual.data[:] = d_pred.data[:] - d_obs.resample(geometry.dt).data[:][0:d_pred.data.shape[0], :]

        # Function value and gradient    
        fval = .5*np.linalg.norm(residual.data.flatten())**2
        solver.gradient(rec=residual, u=u0, vp=model.vp, grad=grad)
        
        # Convert to numpy array and remove absorbing boundaries
        grad_crop = np.array(grad.data[:])[model.nbl:-model.nbl, model.nbl:-model.nbl, model.nbl:-model.nbl]
        # del solver,model,geometry,d_obs,residual,d_pred,u0,grad

        ##clear
        for key, value in globals().items():
            if key == "fval" or "grad_crop":
                continue
            del globals()[key]
            gc.collect()
        return fval, grad_crop

    # Parallel FWI objective function
    def fwi_objective_multi_shots(model, geometry, d_obs):

        futures = []
        for i in range(geometry.nsrc):

            # Geometry for current shot
            geometry_i = AcquisitionGeometry(model, geometry.rec_positions, geometry.src_positions[i,:], 
                geometry.t0, geometry.tn, f0=geometry.f0, src_type=geometry.src_type)
            
            # Call serial FWI objective function for each shot location
            futures.append(client.submit(fwi_objective_single_shot, model, geometry_i, d_obs[i]))

        # Wait for all workers to finish and collect function values and gradients
        wait(futures)
        fval = 0.0
        grad = np.zeros(model.shape)
        for i in range(geometry.nsrc):
            fval += futures[i].result()[0]
            grad += futures[i].result()[1]
        del futures
        gc.collect()
        return fval, grad

    # Wrapper for scipy optimizer: x is current model in squared slowness [s^2/km^2]
    def loss(x, model, geometry, d_obs):
        
        # Convert x to velocity
        v_curr = 1.0/np.sqrt(x.reshape(model.shape))
        
        # Overwrite current velocity in geometry (don't update boundary region)
        model.update('vp', v_curr.reshape(model.shape))
        
        # Evaluate objective function 
        fval, grad = fwi_objective_multi_shots(model, geometry, d_obs)
        # print(sys.getsizeof(v_curr),sys.getsizeof(grad))
        return fval, grad.flatten().astype(np.float64)    # scipy expects double precision vector
    # Start Dask cluster
    start=time.time()
    configuration['log-level'] = 'WARNING'
    # Set up velocity model
    
    spacing = (10., 10. ,10)    # Grid spacing in m. The domain size is now 1km by 1km.
    origin = (0, 0 ,0)         # Need origin to define relative source and receiver locations.
    nbl = 20
    sample=1
    v=sio.loadmat("../../../data/3D_v_model/v0.mat")['v']
    v=v[::sample,::sample,::sample]
    shape = (v.shape[0], v.shape[1], v.shape[2])      # Number of grid points (nx, nz).
    model1 = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                    space_order=6, nbl=nbl, bcs="damp")
    filter_sigma = (10, 10, 10 )
    model0 = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                    space_order=6, nbl=nbl, bcs="damp", grid = model1.grid)
    print("filter_sigma:",filter_sigma)
    gaussian_smooth(model0.vp, sigma=filter_sigma)



    # Set up acquisiton geometry
    t0 = 0.
    tn = 1000. 
    f0 = 0.015

    # Set up source geometry, but define 5 sources instead of just one.
    point_s=5
    nsources = point_s*point_s
    src_coordinates = np.empty((nsources, 3))
    src_coordinates[:, 0] = np.repeat(np.linspace(0, model1.domain_size[0], num=point_s),point_s)
    src_coordinates[:, 1] = np.tile(np.linspace(0, model1.domain_size[0], num=point_s),point_s)
    src_coordinates[:, 2] = 10.# Source depth is 20m

    # Initialize receivers for synthetic and imaging data
    point_r=10
    nreceivers = point_r*point_r
    rec_coordinates = np.empty((nreceivers, 3))
    rec_coordinates[:, 0] = np.repeat(np.linspace(spacing[0], model1.domain_size[0] - spacing[0], num=point_r),point_r)
    rec_coordinates[:, 1] = np.tile(np.linspace(spacing[0], model1.domain_size[0] - spacing[0], num=point_r),point_r) 
    rec_coordinates[:, 2] = 1.# Receiver depth
    # Set up geometry objects for observed and predicted data
    geometry1 = AcquisitionGeometry(model1, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')
    geometry0 = AcquisitionGeometry(model0, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')
    if USE_GPU_AWARE_DASK:
        from dask_cuda import LocalCUDACluster
        cluster = LocalCUDACluster(threads_per_worker=3, death_timeout=600) 
    else:
        from distributed import LocalCluster
        cluster = LocalCluster(n_workers=60, death_timeout=600)
    
    client = Client(cluster)

    # Compute observed data in parallel (inverse crime). In real life we would read the SEG-Y data here.
    d_obs = forward_modeling_multi_shots(model1, geometry1, save=False)

    from examples.seismic import plot_shotrecord

    # Plot shot no. 3 of 5
    plt.figure()
    plot_shotrecord(d_obs[2].data, model1, t0, tn)
    plt.savefig("../../../data/3D_FWI/shot.png")

    from devito import Function
    from examples.seismic import Receiver
    # Compute FWI gradient for 5 shots
    # f, g = fwi_objective_multi_shots(model0, geometry0, d_obs)

    from examples.seismic import plot_image
    # Plot 
    # plt.figure()
    # plot_image(g.reshape(model1.shape)[50], vmin=-6e3, vmax=6e3, cmap="cividis")
    # plt.savefig("../../../data/devito/FWI/test_result/grad_test.png")

    
    # Callback to track model error
    model_error = []
    def fwi_callback(xk):
        m = 1.0 / (model1.vp.data[model1.nbl:-model1.nbl, model1.nbl:-model1.nbl, model1.nbl:-model1.nbl].copy().reshape(-1).astype(np.float64))**2
        model_error.append(np.linalg.norm((xk - m)/m))
        print(time.time()-start,"s")
        if (len(model_error)-1)%1==0:
            plt.figure()
            xk0=1.0/np.sqrt(xk)
            plot_image(xk0.reshape(model1.shape)[3],cmap="cividis")
            plt.savefig("../../../data/3D_FWI/result/v_update{}".format(len(model_error)-1))
            plt.close()
        del m
        gc.collect()

    # Box contraints
    vmin = 1.8    # do not allow velocities slower than water
    vmax = 8.0
    bounds = [(1.0/vmax**2, 1.0/vmin**2) for _ in range(np.prod(model0.shape))]    # in [s^2/km^2]

    # Initial guess
    v0 = model0.vp.data[model0.nbl:-model0.nbl, model0.nbl:-model0.nbl, model0.nbl:-model0.nbl]
    sio.savemat("../../../data/3D_FWI/v_start.mat",{"v":v0})
    plt.figure()
    plot_image(v0[3],cmap="cividis")
    plt.savefig("../../../data/3D_FWI/v_start.png")
    m0 = 1.0 / (v0.reshape(-1).astype(np.float64))**2


    from scipy import optimize

    # FWI with L-BFGS
    
    iter_all=50
    for i in range(iter_all):
        ftol = 0.00001
        maxiter = 1
        result = optimize.minimize(loss, m0, args=(model0, geometry0, d_obs), method='L-BFGS-B', jac=True, 
            callback=fwi_callback, bounds=bounds, options={'ftol':ftol, 'maxiter':maxiter, 'disp':True})
        m0=result["x"].copy()
        # plot v_update
        vp = 1.0/np.sqrt(result['x'].reshape(model1.shape))
        plt.figure()
        plot_image(vp[3],cmap="cividis")
        plt.savefig("../../../data/3D_FWI/v_update.png")
        plt.close()
        # save v_update
        sio.savemat("../../../data/3D_FWI/v_update.mat",{"v":vp})
        # Plot model error
        plt.figure()
        plt.plot(range(1, len(model_error)+1), model_error); plt.xlabel('Iteration number'); plt.ylabel('L2-model error')
        plt.savefig("../../../data/3D_FWI/loss.png")
        plt.close()
        # print time
        end=time.time()
        print(end-start,"s")
        del result,vp
        gc.collect()
        
    # Check termination criteria
    # assert np.isclose(result['fun'], ftol) or result['nit'] == maxiter
    # Plot FWI result
    vp = 1.0/np.sqrt(result['x'].reshape(model1.shape))
    plt.figure()
    plot_image(vp[3],cmap="cividis")
    plt.savefig("../../../data/3D_FWI/v_update.png")
    sio.savemat("../../../data/3D_FWI/v_update.mat",{"v":vp})


    sio.savemat("../../../data/3D_FWI/v_real.mat",{"v":model1.vp.data[model1.nbl:-model1.nbl, model1.nbl:-model1.nbl, model1.nbl:-model1.nbl]})
    plt.figure()
    plot_image(model1.vp.data[model1.nbl:-model1.nbl, model1.nbl:-model1.nbl, model1.nbl:-model1.nbl][3], cmap="cividis")
    plt.savefig("../../../data/3D_FWI/v_real.png")
    

    

    # Plot model error
    plt.figure()
    plt.plot(range(1, len(model_error)+1), model_error); plt.xlabel('Iteration number'); plt.ylabel('L2-model error')
    plt.close()
    plt.savefig("../../../data/3D_FWI/loss.png")
    end=time.time()
    print(end-start,"s")
    print()

    #load data
# v_view=sio.loadmat("../../../data/devito/FWI/test_result/v_update.mat")["v"]
# plt.figure()
# plot_image(v_view[50],cmap="cividis")
# plt.savefig("../../../data/devito/FWI/test_result/v_view.png")