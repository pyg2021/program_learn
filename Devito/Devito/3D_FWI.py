import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
from examples.seismic import Model, plot_velocity
start= time.time()
from devito import configuration
configuration['log-level'] = 'WARNING'


nshots = 36  # Number of shots to create gradient from
nreceivers = 400  # Number of receiver locations per shot 
fwi_iterations = 1  # Number of outer FWI iterations

#NBVAL_IGNORE_OUTPUT
from examples.seismic import demo_model, plot_velocity, plot_perturbation

# Define true and initial model
v=sio.loadmat("/home/pengyaoguang/data/shengli/floded_data/floed_v2.mat")['v']
origin=(0., 0., 0. )
shape=(100, 100, 100)
spacing=(10., 10., 10.)
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                  space_order=8, nbl=20, bcs="damp")

model0 = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                  space_order=8, nbl=20, grid=model.grid,bcs="damp")

from devito import gaussian_smooth
filter_sigma = (1, 1, 1 )
gaussian_smooth(model0.vp, sigma=filter_sigma)
# plot_velocity(model)
# plot_velocity(model0)
# plot_perturbation(model0, model)

plt.figure()
v_update=model0.vp.data[tuple(slice(model0.nbl, -model0.nbl) for _ in range(3))]
plt.imshow(v_update[50].T)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/devito/FWI/start_model.png")
plt.figure()
v_update=model.vp.data[tuple(slice(model.nbl, -model.nbl) for _ in range(3))]
plt.imshow(v_update[50].T)
plt.colorbar()
plt.savefig("/home/pengyaoguang/data/devito/FWI/real_model.png")
assert model.grid == model0.grid
assert model.vp.grid == model0.vp.grid

#NBVAL_IGNORE_OUTPUT
# Define acquisition geometry: source
from examples.seismic import AcquisitionGeometry
t0 = 0.
tn = 1000. 
f0 = 0.010
# First, position source centrally in all dimensions, then set depth
src_coordinates = np.empty((1, 3))
src_coordinates[0, :] = np.array(model.domain_size) * .5
src_coordinates[0, -1] = 20.  # Depth is 20m


# Define acquisition geometry: receivers

# Initialize receivers for synthetic and imaging data
rec_coordinates = np.empty((nreceivers, 3))
rec_coordinates[:, 1] = np.repeat(np.linspace(20, model.domain_size[0], num=20), 20)
rec_coordinates[:, 0] = np.tile(np.linspace(20, model.domain_size[1], num=20), 20)
rec_coordinates[:, 2] = 30.

# Geometry

geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')
# We can plot the time signature to see the wavelet
# geometry.src.show()
# Plot acquisition geometry
# plt.figure()
# plot_velocity(model, source=geometry.src_positions,
#               receiver=geometry.rec_positions[:, :])
# plt.savefig("/home/pengyaoguang/1325/Devito/Devito/result/source_rec_model.png")
# Compute synthetic data with forward operator 
from examples.seismic.acoustic import AcousticWaveSolver

solver = AcousticWaveSolver(model, geometry, space_order=4)
# true_d, _, _ = solver.forward(vp=model.vp)
# # Compute initial data with forward operator 
# smooth_d, _, _ = solver.forward(vp=model0.vp)
#NBVAL_IGNORE_OUTPUT
from examples.seismic import plot_shotrecord

# Plot shot record for true and smooth velocity model and the difference
# plot_shotrecord(true_d.data, model, t0, tn)
# plot_shotrecord(smooth_d.data, model, t0, tn)
# plot_shotrecord(smooth_d.data - true_d.data, model, t0, tn)
#NBVAL_IGNORE_OUTPUT

# Prepare the varying source locations sources
source_locations = np.empty((nshots, 3), dtype=np.float32)

source_locations[:, 0] = np.repeat(np.linspace(20, model.domain_size[0], num=6), 6)
source_locations[:, 1] = np.tile(np.linspace(20, model.domain_size[1], num=6), 6)
source_locations[:, -1] = 30.

# plot_velocity(model, source=source_locations)
from devito import Eq, Operator

# Computes the residual between observed and synthetic data into the residual
def compute_residual(residual, dobs, dsyn):
    if residual.grid.distributor.is_parallel:
        # If we run with MPI, we have to compute the residual via an operator
        # First make sure we can take the difference and that receivers are at the 
        # same position
        assert np.allclose(dobs.coordinates.data[:], dsyn.coordinates.data)
        assert np.allclose(residual.coordinates.data[:], dsyn.coordinates.data)
        # Create a difference operator
        diff_eq = Eq(residual, dsyn.subs({dsyn.dimensions[-1]: residual.dimensions[-1]}) -
                               dobs.subs({dobs.dimensions[-1]: residual.dimensions[-1]}))
        Operator(diff_eq)()
    else:
        # A simple data difference is enough in serial
        residual.data[:] = dsyn.data[:] - dobs.data[:]

    return residual

# Create FWI gradient kernel 
from devito import Function, TimeFunction, norm
from examples.seismic import Receiver

import scipy
def fwi_gradient(vp_in):    
    # Create symbols to hold the gradient
    grad = Function(name="grad", grid=model.grid)
    # Create placeholders for the data residual and data
    residual = Receiver(name='residual', grid=model.grid,
                        time_range=geometry.time_axis, 
                        coordinates=geometry.rec_positions)
    d_obs = Receiver(name='d_obs', grid=model.grid,
                     time_range=geometry.time_axis, 
                     coordinates=geometry.rec_positions)
    d_syn = Receiver(name='d_syn', grid=model.grid,
                     time_range=geometry.time_axis, 
                     coordinates=geometry.rec_positions)
    objective = 0.
    for i in range(nshots):
        # Update source location
        geometry.src_positions[0, :] = source_locations[i, :]

        # Generate synthetic data from true model
        _, _, _ = solver.forward(vp=model.vp, rec=d_obs)

        # Compute smooth data and full forward wavefield u0
        _, u0, _ = solver.forward(vp=vp_in, save=True, rec=d_syn)

        # Compute gradient from data residual and update objective function 
        compute_residual(residual, d_obs, d_syn)

        objective += .5*norm(residual)**2
        solver.gradient(rec=residual, u=u0, vp=vp_in, grad=grad)

    return objective, grad

# Compute gradient of initial model
ff, update = fwi_gradient(model0.vp)
# assert np.isclose(ff, 57283, rtol=1e0)

#NBVAL_IGNORE_OUTPUT
from devito import mmax
from examples.seismic import plot_image

# # Plot the FWI gradient
# plot_image(-update.data, vmin=-1e4, vmax=1e4, cmap="jet")

# # Plot the difference between the true and initial model.
# # This is not known in practice as only the initial model is provided.
# plot_image(model0.vp.data - model.vp.data, vmin=-1e-1, vmax=1e-1, cmap="jet")

# # Show what the update does to the model
# alpha = .5 / mmax(update)
# plot_image(model0.vp.data + alpha*update.data, vmin=2.5, vmax=3.0, cmap="jet")


from sympy import Min, Max
# Define bounding box constraints on the solution.
def update_with_box(vp, alpha, dm, vmin=2.0, vmax=3.5):
    """
    Apply gradient update in-place to vp with box constraint

    Notes:
    ------
    For more advanced algorithm, one will need to gather the non-distributed
    velocity array to apply constrains and such.
    """
    update = vp + alpha * dm
    update_eq = Eq(vp, Max(Min(update, vmax), vmin))
    Operator(update_eq)()

#NBVAL_SKIP

from devito import mmax

# Run FWI with gradient descent
history = np.zeros((fwi_iterations, 1))
for i in range(0, fwi_iterations):
    # Compute the functional value and gradient for the current
    # model estimate
    phi, direction = fwi_gradient(model0.vp)

    # Store the history of the functional values
    history[i] = phi

    # Artificial Step length for gradient descent
    # In practice this would be replaced by a Linesearch (Wolfe, ...)
    # that would guarantee functional decrease Phi(m-alpha g) <= epsilon Phi(m)
    # where epsilon is a minimum decrease constant
    alpha = .05 / mmax(direction)

    # Update the model estimate and enforce minimum/maximum values
    update_with_box(model0.vp , alpha , direction)

    # Log the progress made
    print('Objective value is %f at iteration %d' % (phi, i+1))

    #NBVAL_IGNORE_OUTPUT

    # Plot inverted velocity model
    plt.figure()
    v_update=model0.vp.data[tuple(slice(model.nbl, -model.nbl) for _ in range(3))]
    plt.imshow(v_update[50].T)
    plt.colorbar()
    plt.savefig("/home/pengyaoguang/data/devito/FWI/updated{}_model.png".format(i))
    end=time.time()
    print(end-start,"s")
#NBVAL_SKIP
import matplotlib.pyplot as plt

# Plot objective function decrease
plt.figure()
plt.loglog(history)
plt.xlabel('Iteration number')
plt.ylabel('Misift value Phi')
plt.title('Convergence')
plt.show()
plt.savefig("/home/pengyaoguang/data/devito/FWI/history.png")
sio.savemat("/home/pengyaoguang/data/devito/FWI/v.mat",{"v":model0.vp.data})