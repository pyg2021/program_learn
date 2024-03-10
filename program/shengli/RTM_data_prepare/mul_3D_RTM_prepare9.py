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
for j in range(15000+73,15000+120):
    spacing = (10., 10., 10)  # Grid spacing in m. The domain size is now dx=1km, dy=1km, dz=1km
    origin = (0., 0., 0.)  # What is the location of the top left corner (x,y,z). This is necessary to define
    # Define a velocity profile. The velocity is in km/s
    shape = (100 ,100 ,100 )
    v=sio.loadmat("/home/pengyaoguang/data/3D_v_model/v{}.mat".format(j))['v']
    sample=1
    v=v[::sample,::sample,::sample]
    shape = (v.shape[0], v.shape[1], v.shape[2])  # Number of grid point (nx, ny, nz)
    # Create true model from a preset
    model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                    space_order=6, nbl=20, bcs="damp")

    nshots = 100
    nreceivers = 400
    t0 = 0.
    tn = 1000.  # Simulation last 1 second (1000 ms)
    f0 = 0.015  # Source peak frequency is 10Hz (0.010 kHz)
    #NBVAL_IGNORE_OUTPUT
    from devito import gaussian_smooth

    # Create initial model and smooth the boundaries
    model0 = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                    space_order=4, grid=model.grid, nbl=20, bcs="damp")
    filter_sigma = (10, 10, 10 )
    gaussian_smooth(model0.vp, sigma=filter_sigma)
    # sio.savemat("/home/pengyaoguang/data/3D_v_smooth/v{}_smooth.mat".format(j),{'v':model.vp.data})
    #NBVAL_IGNORE_OUTPUT
    # Define acquisition geometry: source
    from examples.seismic import AcquisitionGeometry

    # First, position source centrally in all dimensions, then set depth
    src_coordinates = np.empty((1, 3))
    src_coordinates[0, :] = np.array(model.domain_size) * .5
    src_coordinates[0, -1] = 10.  # Depth is 20m


    # Define acquisition geometry: receivers

    # Initialize receivers for synthetic and imaging data
    rec_coordinates = np.empty((nreceivers, 3))
    rec_coordinates[:, 0] = np.repeat(np.linspace(0, model.domain_size[0], num=20), 20)
    rec_coordinates[:, 1] = np.tile(np.linspace(20, model.domain_size[1], num=20), 20)
    # rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nreceivers)
    # rec_coordinates[:, 1] = model.domain_size[1]*0.5
    rec_coordinates[:, 2] = 1.

    # Geometry
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=.010, src_type='Ricker')
    # We can plot the time signature to see the wavelet

    # Compute synthetic data with forward operator 
    from examples.seismic.acoustic import AcousticWaveSolver

    solver = AcousticWaveSolver(model, geometry, space_order=4)

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
    source_locations[:, 0] = np.repeat(np.linspace(0., model.domain_size[0], num=10),10)
    source_locations[:, 1] = np.tile(np.linspace(0., model.domain_size[1], num=10),10)
    # source_locations[:, 0] = np.linspace(0., 1000, num=nshots)
    # source_locations[:, 1] = model.domain_size[1]*0.5
    source_locations[:, 2] = 1.
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
        sio.savemat("/home/pengyaoguang/data/3D_RTM/RTM{}.mat".format(j),{"RTM":image.data})


