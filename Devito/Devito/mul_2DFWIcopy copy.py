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
                space_order=8, nbl=nbl, bcs="damp")
filter_sigma = (2, 2, 2 )
model0 = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                space_order=4, nbl=nbl, bcs="damp", grid = model1.grid)
print("filter_sigma:",filter_sigma)
gaussian_smooth(model0.vp, sigma=filter_sigma)


# model1 = demo_model('layers-isotropic', 
#     origin=origin, shape=shape, spacing=spacing, nbl=nbl,nlayers=5)

# # Initial model
# model0 = demo_model('layers-isotropic', vp_circle=2.5, vp_background=2.5,
#     origin=origin, shape=shape, spacing=spacing, nbl=nbl,nlayers=5, grid = model1.grid)

from examples.seismic import AcquisitionGeometry
import numpy as np

# Set up acquisiton geometry
t0 = 0.
tn = 1000. 
f0 = 0.010

# Set up source geometry, but define 5 sources instead of just one.
nsources = 1
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

geometry_i = AcquisitionGeometry(model1, geometry1.rec_positions, geometry1.src_positions[0,:], 
    geometry1.t0, geometry1.tn, f0=geometry1.f0, src_type=geometry1.src_type)
        
a,b=forward_modeling_single_shot(model1, geometry_i, save=False, dt=4.0)
print()