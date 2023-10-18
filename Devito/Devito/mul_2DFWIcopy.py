USE_GPU_AWARE_DASK = False  

from examples.seismic import demo_model,Model
import matplotlib.pyplot as plt
from devito import configuration
import time
import scipy.io as sio
from devito import gaussian_smooth
start=time.time()
configuration['log-level'] = 'WARNING'
# Set up velocity model
shape = (30, 30, 30)      # Number of grid points (nx, nz).
spacing = (10., 10., 10.)    # Grid spacing in m. The domain size is now 1km by 1km.
origin = (0, 0, 0)         # Need origin to define relative source and receiver locations.
nbl = 20

v=sio.loadmat("/home/pengyaoguang/data/shengli/data_all/floed_v0.mat")['v']
v=v[:30,:30,:30]
model1 = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                space_order=2, nbl=nbl, bcs="damp")
filter_sigma = (2, 2, 2 )
model0 = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                space_order=2, nbl=nbl, bcs="damp", grid = model1.grid)
print("filter_sigma:",filter_sigma)
gaussian_smooth(model0.vp, sigma=filter_sigma)


from examples.seismic import AcquisitionGeometry
import numpy as np

# Set up acquisiton geometry
t0 = 0.
tn = 1000. 
f0 = 0.010

# Set up source geometry, but define 5 sources instead of just one.
nsources = 5
src_coordinates = np.empty((nsources, 3))
src_coordinates[:, 1] = np.linspace(0, model1.domain_size[0], num=nsources)
src_coordinates[:, 0] = 20. 
src_coordinates[:, 2] = 20.  # Source depth is 20m

# Initialize receivers for synthetic and imaging data
nreceivers = 101
rec_coordinates = np.empty((nreceivers, 3))
rec_coordinates[:, 1] = np.linspace(spacing[0], model1.domain_size[0] - spacing[0], num=nreceivers)
rec_coordinates[:, 0] = 20.   
rec_coordinates[:, 2] = 20.  # Receiver depth
# Set up geometry objects for observed and predicted data
geometry1 = AcquisitionGeometry(model1, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')
geometry0 = AcquisitionGeometry(model0, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')

from examples.seismic.acoustic import AcousticWaveSolver

# Serial modeling function
def forward_modeling_single_shot(model, geometry, save=False, dt=4.0):
    solver = AcousticWaveSolver(model, geometry, space_order=4)
    d_obs, u0 = solver.forward(vp=model.vp, save=save)[0:2]
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

from distributed import Client, wait

# Start Dask cluster
if __name__ == "__main__":
    if USE_GPU_AWARE_DASK:
        from dask_cuda import LocalCUDACluster
        cluster = LocalCUDACluster(threads_per_worker=2, death_timeout=600) 
    else:
        from distributed import LocalCluster
        cluster = LocalCluster(n_workers=nsources, death_timeout=600)
    
    client = Client(cluster)

    # Compute observed data in parallel (inverse crime). In real life we would read the SEG-Y data here.
    d_obs = forward_modeling_multi_shots(model1, geometry1, save=False)

    from examples.seismic import plot_shotrecord

    # Plot shot no. 3 of 5
    # plt.figure()
    # plot_shotrecord(d_obs[2].data, model1, t0, tn)
    # plt.savefig("/home/pengyaoguang/1325/Devito/Devito/result/mul_FWI.png")

    from devito import Function
    from examples.seismic import Receiver

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
        grad_crop = np.array(grad.data[:])[model.nbl:-model.nbl, model.nbl:-model.nbl,model.nbl:-model.nbl]
        
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

        return fval, grad


    # Compute FWI gradient for 5 shots
    f, g = fwi_objective_multi_shots(model0, geometry0, d_obs)

    from examples.seismic import plot_image
    # Plot g
    # plot_image(g.reshape(model1.shape), vmin=-6e3, vmax=6e3, cmap="cividis")
    # Wrapper for scipy optimizer: x is current model in squared slowness [s^2/km^2]
    def loss(x, model, geometry, d_obs):
        
        # Convert x to velocity
        v_curr = 1.0/np.sqrt(x.reshape(model.shape))
        
        # Overwrite current velocity in geometry (don't update boundary region)
        model.update('vp', v_curr.reshape(model.shape))
        
        # Evaluate objective function 
        fval, grad = fwi_objective_multi_shots(model, geometry, d_obs)
        return fval, grad.flatten().astype(np.float64)    # scipy expects double precision vector
    # Callback to track model error
    model_error = []
    def fwi_callback(xk):
        vp = model1.vp.data[model1.nbl:-model1.nbl, model1.nbl:-model1.nbl,model1.nbl:-model1.nbl]
        m = 1.0 / (vp.reshape(-1).astype(np.float64))**2
        model_error.append(np.linalg.norm((xk - m)/m))

    # Box contraints
    vmin = 1.4    # do not allow velocities slower than water
    vmax = 4.0
    bounds = [(1.0/vmax**2, 1.0/vmin**2) for _ in range(np.prod(model0.shape))]    # in [s^2/km^2]

    # Initial guess
    v0 = model0.vp.data[model0.nbl:-model0.nbl, model0.nbl:-model0.nbl,model1.nbl:-model1.nbl]
    m0 = 1.0 / (v0.reshape(-1).astype(np.float64))**2


    from scipy import optimize

    # FWI with L-BFGS
    ftol = 0.1
    maxiter = 1
    result = optimize.minimize(loss, m0, args=(model0, geometry0, d_obs), method='L-BFGS-B', jac=True, 
        callback=fwi_callback, bounds=bounds, options={'ftol':ftol, 'maxiter':maxiter, 'disp':True})

    # Check termination criteria
    assert np.isclose(result['fun'], ftol) or result['nit'] == maxiter

    # Plot FWI result
    vp = 1.0/np.sqrt(result['x'].reshape(model1.shape))
    # plot_image(model1.vp.data[model1.nbl:-model1.nbl, model1.nbl:-model1.nbl,model1.nbl:-model1.nbl], vmin=2.4, vmax=2.8, cmap="cividis")
    # plot_image(vp, vmin=2.4, vmax=2.8, cmap="cividis")

    import matplotlib.pyplot as plt

    # Plot model error
    # plt.plot(range(1, maxiter+1), model_error); plt.xlabel('Iteration number'); plt.ylabel('L2-model error')
    # plt.show()

    print(time.time()-start,"s")