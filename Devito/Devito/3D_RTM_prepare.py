import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from examples.seismic import Model, plot_velocity
from devito import configuration
configuration['log-level'] = 'WARNING'
# Configure model presets
from examples.seismic import demo_model
import time
import scipy.io as sio

start=time.time()
# Enable model presets here:
preset = 'layers-isotropic'  # A simple but cheap model (recommended)
# Standard preset with a simple two-layer model
if preset == 'layers-isotropic':
    def create_model(grid=None):
        return demo_model('layers-isotropic', origin=(0., 0., 0. ), shape=(101, 101, 101),
                          spacing=(10., 10., 10.), nbl=20, grid=grid, nlayers=2)

shape = (100, 100, 100)  # Number of grid point (nx, ny, nz)
spacing = (10., 10., 10)  # Grid spacing in m. The domain size is now dx=1km, dy=1km, dz=1km
origin = (0., 0., 0.)  # What is the location of the top left corner (x,y,z). This is necessary to define
 # Define a velocity profile. The velocity is in km/s
# v = np.empty(shape, dtype=np.float32)
# v[:, :, :51] = 1.5
# v[:, :, 51:] = 2.5
v=sio.loadmat("/home/pengyaoguang/data/shengli/data_all/floed_v0.mat")['v']
# Create true model from a preset
plt.figure()
plt.imshow(v[:,70,:])
plt.colorbar()
plt.savefig("/home/pengyaoguang/1325/Devito/Devito/result/8.png")
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                  space_order=8, nbl=20, bcs="damp")

nshots = 400
nreceivers = 400
t0 = 0.
tn = 1000.  # Simulation last 1 second (1000 ms)
f0 = 0.015  # Source peak frequency is 10Hz (0.010 kHz)
#NBVAL_IGNORE_OUTPUT
from devito import gaussian_smooth

# Create initial model and smooth the boundaries
model0 = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                  space_order=8, grid=model.grid, nbl=50, bcs="damp")
filter_sigma = (1, 1, 1 )
gaussian_smooth(model0.vp, sigma=filter_sigma)

#NBVAL_IGNORE_OUTPUT
# Define acquisition geometry: source
from examples.seismic import AcquisitionGeometry

# First, position source centrally in all dimensions, then set depth
src_coordinates = np.empty((1, 3))
src_coordinates[0, :] = np.array(model.domain_size) * .5
src_coordinates[0, -1] = 20.  # Depth is 20m


# Define acquisition geometry: receivers

# Initialize receivers for synthetic and imaging data
rec_coordinates = np.empty((nreceivers, 3))
rec_coordinates[:, 0] = np.repeat(np.linspace(0, model.domain_size[0], num=20), 20)
rec_coordinates[:, 1] = np.tile(np.linspace(20, model.domain_size[1], num=20), 20)
# rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nreceivers)
# rec_coordinates[:, 1] = model.domain_size[1]*0.5
rec_coordinates[:, 2] = 30.

# Geometry
geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=.010, src_type='Ricker')
# We can plot the time signature to see the wavelet
plt.figure()
geometry.src.show()
plt.savefig("/home/pengyaoguang/1325/Devito/Devito/result/4.png")

# Compute synthetic data with forward operator 
from examples.seismic.acoustic import AcousticWaveSolver

solver = AcousticWaveSolver(model, geometry, space_order=4)
# true_d , _, _ = solver.forward(vp=model.vp)
# # Compute initial data with forward operator 
# smooth_d, _, _ = solver.forward(vp=model0.vp)
#NBVAL_IGNORE_OUTPUT
# Plot shot record for true and smooth velocity model and the difference
# from examples.seismic import plot_shotrecord

# plot_shotrecord(true_d.data, model, t0, tn)
# plot_shotrecord(smooth_d.data, model, t0, tn)
# plot_shotrecord(smooth_d.data - true_d.data, model, t0, tn)

# Define gradient operator for imaging
from devito import TimeFunction, Operator, Eq, solve
from examples.seismic import PointSource

def ImagingOperator(model, image):
    # Define the wavefield with the size of the model and the time dimension
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)

    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=4,
                     save=geometry.nt)

    # Define the wave equation, but with a negated damping term
    eqn = model.m * v.dt2 - v.laplace + model.damp * v.dt.T

    # Use `solve` to rearrange the equation into a stencil expression
    stencil = Eq(v.backward, solve(eqn, v.backward))

    # Define residual injection at the location of the forward receivers
    dt = model.critical_dt
    residual = PointSource(name='residual', grid=model.grid,
                           time_range=geometry.time_axis,
                           coordinates=geometry.rec_positions)    
    res_term = residual.inject(field=v.backward, expr=residual * dt**2 / model.m)

    # Correlate u and v for the current time step and add it to the image
    image_update = Eq(image, image - u * v)

    return Operator([stencil] + res_term + [image_update],
                    subs=model.spacing_map)

#NBVAL_IGNORE_OUTPUT

# Prepare the varying source locations
source_locations = np.empty((nshots, 3), dtype=np.float32)
source_locations[:, 0] = np.repeat(np.linspace(0., 1000, num=20),20)
source_locations[:, 1] = np.tile(np.linspace(0., 1000, num=20),20)
# source_locations[:, 0] = np.linspace(0., 1000, num=nshots)
# source_locations[:, 1] = model.domain_size[1]*0.5
source_locations[:, 2] = 30.
# plt.figure()
# plot_velocity(model, source=source_locations)
# plt.savefig("/home/pengyaoguang/1325/Devito/Devito/result/5.png")


# Run imaging loop over shots
from devito import Function

# Create image symbol and instantiate the previously defined imaging operator
image = Function(name='image', grid=model.grid)
op_imaging = ImagingOperator(model, image)

for i in range(nshots):
    print('Imaging source %d out of %d' % (i+1, nshots))

    # Update source location
    geometry.src_positions[0, :] = source_locations[i, :]

    # Generate synthetic data from true model
    true_d, _, _ = solver.forward(vp=model.vp)

    # Compute smooth data and full forward wavefield u0
    smooth_d, u0, _ = solver.forward(vp=model0.vp, save=True)

    # Compute gradient from the data residual  
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)
    residual = smooth_d.data - true_d.data
    op_imaging(u=u0, v=v, vp=model0.vp, dt=model0.critical_dt, 
               residual=residual)
    end=time.time()
    print(end-start,"s")
#NBVAL_IGNORE_OUTPUT

from examples.seismic import plot_image

# Plot the inverted image
plt.figure()
plot_image(np.diff(image.data, axis=1)[:,70,:])
plt.savefig("/home/pengyaoguang/1325/Devito/Devito/result/6.png")
sio.savemat("/home/pengyaoguang/data/shengli/data_all_RTM/RTM_easy.mat",{"RTM":image.data})

from examples.seismic import plot_image
plt.figure()
data=np.diff(sio.loadmat("/home/pengyaoguang/data/shengli/data_all_RTM/RTM_easy.mat")["RTM"], axis=1)
max=np.max(data)
plt.imshow(data[50,:,:].T/max,vmax=0.1,vmin=-0.1,cmap="gray")
plt.colorbar()
plt.savefig("/home/pengyaoguang/1325/Devito/Devito/result/6.png")

