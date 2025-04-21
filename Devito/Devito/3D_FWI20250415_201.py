import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
from examples.seismic import Model, plot_velocity
start= time.time()
from devito import configuration
configuration['log-level'] = 'WARNING'


nshots = 100  # Number of shots to create gradient from
nreceivers = 400  # Number of receiver locations per shot 
fwi_iterations = 1000  # Number of outer FWI iterations
print("some information:\n","nshots:",nshots,"nreceivers:",nreceivers,"fwi_iterations:",fwi_iterations)
#NBVAL_IGNORE_OUTPUT
from examples.seismic import demo_model, plot_velocity, plot_perturbation

m=201
n=50
# Define true and initial model
v=sio.loadmat("/home/yaoguang/data//3D_RTM2/v{}.mat".format(m-1))['v']
# v=sio.loadmat("/home/yaoguang/data/3D_v_model/v{}.mat".format(m))['v']
origin=(0., 0., 0. )
shape=(100, 100, 100)
spacing=(10., 10., 10.)
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                  space_order=6, nbl=20, bcs="damp")

model0 = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                  space_order=6, nbl=20, grid=model.grid,bcs="damp")

from devito import gaussian_smooth
filter_sigma = (30, 30, 30 )
print("filter_sigma:",filter_sigma)
gaussian_smooth(model0.vp, sigma=filter_sigma)
# plot_velocity(model)
# plot_velocity(model0)
# plot_perturbation(model0, model)
# model0.vp.data[:]=sio.loadmat("/home/yaoguang/data/3D_FWI/v_update{}_{}.mat".format(m,n))["v"]
# print("111111")
# sio.savemat("/home/yaoguang/data/3D_FWI/v_start{}.mat".format(m),{"v":model0.vp.data})
plt.figure()
v_update=model0.vp.data[tuple(slice(model0.nbl, -model0.nbl) for _ in range(3))]
plt.imshow(v_update[n].T,vmin=1.8,vmax=6,cmap='jet')
plt.colorbar()
plt.savefig("/home/yaoguang/data/3D_FWI/v_start{}_{}.png".format(m,n))
plt.close()
plt.figure()
v_update=model.vp.data[tuple(slice(model.nbl, -model.nbl) for _ in range(3))]
plt.imshow(v_update[n].T,vmin=1.8,vmax=6)
plt.colorbar()
plt.savefig("/home/yaoguang/data/3D_FWI/v_real{}_{}.png".format(m,n))
plt.close()
assert model.grid == model0.grid
assert model.vp.grid == model0.vp.grid

#NBVAL_IGNORE_OUTPUT
# Define acquisition geometry: source
from examples.seismic import AcquisitionGeometry
t0 = 0.
tn = 1000. 
f0 = 0.015
# First, position source centrally in all dimensions, then set depth
src_coordinates = np.empty((1, 3))
src_coordinates[0, :] = np.array(model.domain_size) * .5
src_coordinates[0, -1] = 10.  # Depth is 20m


# Define acquisition geometry: receivers

# Initialize receivers for synthetic and imaging data
rec_coordinates = np.empty((nreceivers, 3))
rec_coordinates[:, 1] = np.repeat(np.linspace(20, model.domain_size[0], num=20), 20)
rec_coordinates[:, 0] = np.tile(np.linspace(20, model.domain_size[1], num=20), 20)
rec_coordinates[:, 2] = 1.

# Geometry

geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')
# We can plot the time signature to see the wavelet
# geometry.src.show()
# Plot acquisition geometry
# plt.figure()
# plot_velocity(model, source=geometry.src_positions,
#               receiver=geometry.rec_positions[:, :])
# plt.savefig("/home/yaoguang/1325/Devito/Devito/result/source_rec_model.png")
# Compute synthetic data with forward operator 
from examples.seismic.acoustic import AcousticWaveSolver

solver = AcousticWaveSolver(model, geometry, space_order=6)
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

source_locations[:, 0] = np.repeat(np.linspace(20, model.domain_size[0], num=10), 10)
source_locations[:, 1] = np.tile(np.linspace(20, model.domain_size[1], num=10), 10)
source_locations[:, -1] = 1.

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
        print(i)
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
# ff, update = fwi_gradient(model0.vp)
# assert np.isclose(ff, 57283, rtol=1e0)

# #NBVAL_IGNORE_OUTPUT
# from devito import mmax
# from examples.seismic import plot_image

# # Plot the FWI gradient
# plt.figure()
# plot_image(-update.data[50], vmin=-1e4, vmax=1e4, cmap="jet")
# plt.savefig("/home/yaoguang/data/devito/FWI/test_update_data.png")
# # Plot the difference between the true and initial model.
# # This is not known in practice as only the initial model is provided.
# plt.figure()
# plot_image(model0.vp.data[50] - model.vp.data[50], vmin=-1e-1, vmax=1e-1, cmap="jet")
# plt.savefig("/home/yaoguang/data/devito/FWI/test_v_error.png")
# # Show what the update does to the model
# plt.figure()
# alpha = .5 / mmax(update)
# plot_image(model0.vp.data[50][20:120,20:120] + alpha*update.data[50][20:120,20:120],cmap="jet")
# plt.savefig("/home/yaoguang/data/devito/FWI/test_v_update.png")


from sympy import Min, Max
# Define bounding box constraints on the solution.
def update_with_box(vp, alpha, dm, vmin=1.5, vmax=10):
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
    plt.imshow(v_update[n].T,vmin=1.5,vmax=6)
    plt.colorbar()
    plt.savefig("/home/yaoguang/data/3D_FWI/v_update{}_{}model.png".format(m,n))
    sio.savemat("/home/yaoguang/data/3D_FWI/v_update{}_{}.mat".format(m,n),{"v":model0.vp.data})
    plt.close()
    print("loss:",phi)
    end=time.time()
    print(end-start,"s")

    # Plot objective function decrease
    plt.figure()
    plt.loglog(history)
    plt.xlabel('Iteration number')
    plt.ylabel('Misift value Phi')
    plt.title('Convergence')
    # plt.show()
    plt.savefig("/home/yaoguang/data/3D_FWI/history{}_{}.png".format(m,n))
    sio.savemat("/home/yaoguang/data/3D_FWI/history{}_{}.mat".format(m,n),{'h':history})
    plt.close()
# sio.savemat("/home/yaoguang/data/3D_FWI/v1.mat",{"v":model0.vp.data})