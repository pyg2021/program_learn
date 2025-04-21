from examples.seismic import demo_model
import  matplotlib.pyplot as plt
USE_GPU_AWARE_DASK = False
# Set up velocity model
shape = (101, 101)      # Number of grid points (nx, nz).
spacing = (10., 10.)    # Grid spacing in m. The domain size is now 1km by 1km.
origin = (0, 0)         # Need origin to define relative source and receiver locations.
nbl = 40

# True model
model1 = demo_model('circle-isotropic', vp_circle=3.0, vp_background=2.5,
    origin=origin, shape=shape, spacing=spacing, nbl=nbl)

# Initial model
model0 = demo_model('circle-isotropic', vp_circle=2.5, vp_background=2.5,
    origin=origin, shape=shape, spacing=spacing, nbl=nbl, grid = model1.grid)


from examples.seismic import AcquisitionGeometry
import numpy as np

# Set up acquisiton geometry
t0 = 0.
tn = 1000. 
f0 = 0.010

# Set up source geometry, but define 5 sources instead of just one.
nsources = 5
src_coordinates = np.empty((nsources, 2))
src_coordinates[:, 1] = np.linspace(0, model1.domain_size[0], num=nsources)
src_coordinates[:, 0] = 20.  # Source depth is 20m

# Initialize receivers for synthetic and imaging data
nreceivers = 101
rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 1] = np.linspace(spacing[0], model1.domain_size[0] - spacing[0], num=nreceivers)
rec_coordinates[:, 0] = 980.    # Receiver depth
# Set up geometry objects for observed and predicted data
geometry1 = AcquisitionGeometry(model1, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')
geometry0 = AcquisitionGeometry(model0, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')

from examples.seismic.acoustic import AcousticWaveSolver

# Serial modeling function
def forward_modeling_single_shot(model, geometry, save=False, dt=4.0):
    print(2)
    solver = AcousticWaveSolver(model, geometry, space_order=4)
    d_obs, u0 = solver.forward(vp=model.vp, save=save)[0:2]
    return d_obs.resample(dt), u0

# Parallel modeling function
def forward_modeling_multi_shots(model, geometry, save=False, dt=4.0):
    

    futures = []
    for i in range(geometry.nsrc):
        print(1)
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
if USE_GPU_AWARE_DASK:
    from dask_cuda import LocalCUDACluster
    cluster = LocalCUDACluster(threads_per_worker=1, death_timeout=600) 
else:
    from distributed import LocalCluster
    cluster = LocalCluster(n_workers=nsources, death_timeout=600)
    
client = Client(cluster)

# Compute observed data in parallel (inverse crime). In real life we would read the SEG-Y data here.
d_obs = forward_modeling_multi_shots(model1, geometry1, save=False)

from examples.seismic import plot_shotrecord

# Plot shot no. 3 of 5
plot_shotrecord(d_obs[2].data, model1, t0, tn)
plt.savefig("/home/yaoguang/program_learn/Devito/Devito/result/0.png")
