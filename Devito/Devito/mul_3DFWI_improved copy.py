# Set up inversion parameters.
param = {'t0': 0.,
         'tn': 1000.,              # Simulation last 1 second (1000 ms)
         'f0': 0.010,              # Source peak frequency is 10Hz (0.010 kHz)
         'nshots': 5,              # Number of shots to create gradient from
         'shape': (30, 30, 30),      # Number of grid points (nx, nz).
         'spacing': (10., 10., 10.),    # Grid spacing in m. The domain size is now 1km by 1km.
         'origin': (0, 0, 0),         # Need origin to define relative source and receiver locations.
         'nbl': 10,
         'filter_sigma' :(20, 20, 20 )}                # nbl thickness.

import numpy as np
import time
import scipy.io as sio
from scipy import signal, optimize
import matplotlib.pyplot as plt
from devito import Grid
from devito import gaussian_smooth
from distributed import Client, LocalCluster, wait

import cloudpickle as pickle

# Import acoustic solver, source and receiver modules.
from examples.seismic import Model, demo_model, AcquisitionGeometry, Receiver
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import AcquisitionGeometry

# Import convenience function for plotting results
from examples.seismic import plot_image
from examples.seismic import plot_shotrecord

start=time.time()
def get_true_model():
    ''' Define the test phantom; in this case we are using
    a simple circle so we can easily see what is going on.
    '''
    v=sio.loadmat("/home/pengyaoguang/data/shengli/data_all/floed_v0.mat")['v']
    v=v[:30,:30,:30]
    return Model(vp=v, origin=param['origin'], shape=param['shape'], spacing=param['spacing'],
                  space_order=6, nbl=param['nbl'], bcs="damp")

def get_initial_model():
    '''The initial guess for the subsurface model.
    '''
    # Make sure both model are on the same grid
    grid = get_true_model().grid
    v=sio.loadmat("/home/pengyaoguang/data/shengli/data_all/salt_v0.mat")['v']
    v=v[:30,:30,:30]
    M0=Model(vp=v, origin=param['origin'], shape=param['shape'], spacing=param['spacing'],
                    space_order=4, nbl=param['nbl'], bcs="damp",grid=grid)
    gaussian_smooth(M0.vp, sigma=param['filter_sigma'])
    return M0

def wrap_model(x, astype=None):
    '''Wrap a flat array as a subsurface model.
    '''
    model = get_initial_model()
    v_curr = 1.0/np.sqrt(x.reshape(model.shape))
    
    if astype:
        model.update('vp', v_curr.astype(astype).reshape(model.shape))
    else:
        model.update('vp', v_curr.reshape(model.shape))
    return model

def load_model(filename):
    """ Returns the current model. This is used by the
    worker to get the current model.
    """
    pkl = pickle.load(open(filename, "rb"))
    
    return pkl['model']

def dump_model(filename, model):
    ''' Dump model to disk.
    '''
    pickle.dump({'model':model}, open(filename, "wb"))
    
def load_shot_data(shot_id, dt):
    ''' Load shot data from disk, resampling to the model time step.
    '''
    pkl = pickle.load(open("/home/pengyaoguang/data/devito/FWI/improved_FWI/shot_%d.p"%shot_id, "rb"))
    
    return pkl['geometry'], pkl['rec'].resample(dt)

def dump_shot_data(shot_id, rec, geometry):
    ''' Dump shot data to disk.
    '''
    pickle.dump({'rec':rec, 'geometry': geometry}, open('/home/pengyaoguang/data/devito/FWI/improved_FWI/shot_%d.p'%shot_id, "wb"))
    
def generate_shotdata_i(param):
    """ Inversion crime alert! Here the worker is creating the
        'observed' data using the real model. For a real case
        the worker would be reading seismic data from disk.
    """
    # Reconstruct objects
    with open("/home/pengyaoguang/data/devito/FWI/improved_FWI/arguments.pkl", "rb") as cp_file:
        cp = pickle.load(cp_file)
        
    solver = cp['solver']

    # source position changes according to the index
    shot_id=param['shot_id']
    
    solver.geometry.src_positions[0,:]=[20, shot_id*1000./(param['nshots']-1),20]
    model=get_true_model()
    model0=get_initial_model()
    true_d = solver.forward(vp=model0.vp)[0]
    dump_shot_data(shot_id, true_d.resample(4.0), solver.geometry.src_positions)

def generate_shotdata(solver):
    # Pick devito objects (save on disk)
    cp = {'solver': solver}
    with open("/home/pengyaoguang/data/devito/FWI/improved_FWI/arguments.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file) 

    work = [dict(param) for i in range(param['nshots'])]
    # synthetic data is generated here twice: serial(loop below) and parallel (via dask map functionality) 
    for i in  range(param['nshots']):
        work[i]['shot_id'] = i
        generate_shotdata_i(work[i])

    # Map worklist to cluster, We pass our function and the dictionary to the map() function of the client
    # This returns a list of futures that represents each task
    futures = c.map(generate_shotdata_i, work)

    # Wait for all futures
    wait(futures)


# Define a type to store the functional and gradient.
class fg_pair:
    def __init__(self, f, g):
        self.f = f
        self.g = g
    
    def __add__(self, other):
        f = self.f + other.f
        g = self.g + other.g
        
        return fg_pair(f, g)
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
#NBVAL_IGNORE_OUTPUT
from devito import Function

# Create FWI gradient kernel for a single shot
def fwi_gradient_i(param):

    # Load the current model and the shot data for this worker.
    # Note, unlike the serial example the model is not passed in
    # as an argument. Broadcasting large datasets is considered
    # a programming anti-pattern and at the time of writing it
    # it only worked reliably with Dask master. Therefore, the
    # the model is communicated via a file.
    model0 = load_model(param['model'])
    
    dt = model0.critical_dt
    nbl = model0.nbl

    # Get src_position and data
    src_positions, rec = load_shot_data(param['shot_id'], dt)

    # Set up solver -- load the solver used above in the generation of the syntethic data.    
    with open("/home/pengyaoguang/data/devito/FWI/improved_FWI/arguments.pkl", "rb") as cp_file:
        cp = pickle.load(cp_file)
    solver = cp['solver']

    # Set attributes to solver
    solver.geometry.src_positions=src_positions
    solver.geometry.resample(dt)

    # Compute simulated data and full forward wavefield u0

    # d, u0 = solver.forward(vp=model0.vp,save=True)[0:2]
    d, u0 = solver.forward(vp=model0.vp, dt=dt, save=True)[0:2]
    
    # Compute the data misfit (residual) and objective function
    residual = Receiver(name='rec', grid=model0.grid,
                        time_range=solver.geometry.time_axis,
                        coordinates=solver.geometry.rec_positions)

    #residual.data[:] = d.data[:residual.shape[0], :] - rec.data[:residual.shape[0], :]
    residual.data[:] = d.data[:] - rec.data[0:d.data.shape[0], :]
    f = .5*np.linalg.norm(residual.data.flatten())**2

    # Compute gradient using the adjoint-state method. Note, this
    # backpropagates the data misfit through the model.
    grad = Function(name="grad", grid=model0.grid)
    solver.gradient(rec=residual, u=u0, vp=model0.vp, dt=dt, grad=grad)
    
    # Copying here to avoid a (probably overzealous) destructor deleting
    # the gradient before Dask has had a chance to communicate it.
    g = np.array(grad.data[:])[nbl:-nbl, nbl:-nbl, nbl:-nbl]    
    
    # return the objective functional and gradient.
    return fg_pair(f, g)

def fwi_gradient(model, param):
    # Dump a copy of the current model for the workers
    # to pick up when they are ready.
    param['model'] = "/home/pengyaoguang/data/devito/FWI/improved_FWI/model_0.p"
    dump_model(param['model'], wrap_model(model))

    # Define work list
    work = [dict(param) for i in range(param['nshots'])]
    for i in  range(param['nshots']):
        work[i]['shot_id'] = i
        
    # Distribute worklist to workers.
    fgi = c.map(fwi_gradient_i, work, retries=1)
    
    # Perform reduction.
    fg = c.submit(sum, fgi).result()
    
    # L-BFGS in scipy expects a flat array in 64-bit floats.
    return fg.f, fg.g.flatten().astype(np.float64)



if __name__=="__main__":

    #NBVAL_IGNORE_OUTPUT
    from examples.seismic import plot_shotrecord

    # Client setup
    cluster = LocalCluster(n_workers=5, death_timeout=600)
    c = Client(cluster)

    # Generate shot data.
    true_model = get_true_model()
    # Source coords definition
    src_coordinates = np.empty((1, len(param['shape'])))
    # Number of receiver locations per shot.
    nreceivers = 101
    # Set up receiver data and geometry.
    rec_coordinates = np.empty((nreceivers, len(param['shape'])))
    rec_coordinates[:, 0] = np.linspace(param['spacing'][0], true_model.domain_size[0] - param['spacing'][0], num=nreceivers)
    rec_coordinates[:, 1] = 20. 
    rec_coordinates[:, 2] = 20.# 20m from the right end
    # Geometry 
    geometry = AcquisitionGeometry(true_model, rec_coordinates, src_coordinates,
                                param['t0'], param['tn'], src_type='Ricker',
                                f0=param['f0'])
    # Set up solver
    solver = AcousticWaveSolver(true_model, geometry, space_order=4)
    # generate_shotdata(solver)
    print(time.time()-start,"s")



    from scipy import optimize

    # Many optimization methods in scipy.optimize.minimize accept a callback
    # function that can operate on the solution after every iteration. Here
    # we use this to monitor the true relative solution error.
    relative_error = []
    def fwi_callbacks(x):    
        # Calculate true relative error
        true_vp = get_true_model().vp.data[param['nbl']:-param['nbl'], param['nbl']:-param['nbl'],param['nbl']:-param['nbl']]
        true_m = 1.0 / (true_vp.reshape(-1).astype(np.float64))**2
        relative_error.append(np.linalg.norm((x-true_m)/true_m))

    # FWI with L-BFGS
    ftol = 0.0001
    maxiter = 1

    def fwi(model, param, ftol=ftol, maxiter=maxiter):
        # Initial guess
        v0 = model.vp.data[param['nbl']:-param['nbl'], param['nbl']:-param['nbl'],param['nbl']:-param['nbl']]
        m0 = 1.0 / (v0.reshape(-1).astype(np.float64))**2
        
        # Define bounding box constraints on the solution.
        vmin = 1.4    # do not allow velocities slower than water
        vmax = 8.0
        bounds = [(1.0/vmax**2, 1.0/vmin**2) for _ in range(np.prod(model.shape))]    # in [s^2/km^2]
        
        result = optimize.minimize(fwi_gradient,
                                m0, args=(param, ), method='L-BFGS-B', jac=True,
                                bounds=bounds, callback=fwi_callbacks,
                                options={'ftol':ftol,
                                            'maxiter':maxiter,
                                            'disp':True})

        return result
    
    #NBVAL_IGNORE_OUTPUT

    model0 = get_initial_model()
    # model0=get_true_model()
    plt.figure()
    plot_image(true_model.vp.data[10],cmap="cividis")
    plt.savefig("/home/pengyaoguang/data/devito/FWI/test_result/improved_v_start.png")
    # Baby steps
    # result = fwi(model0, param)
    dt = model0.critical_dt
    nbl = model0.nbl

    # Get src_position and data
    src_positions, rec = load_shot_data(0, dt)

    # Set up solver -- load the solver used above in the generation of the syntethic data.    
    with open("/home/pengyaoguang/data/devito/FWI/improved_FWI/arguments.pkl", "rb") as cp_file:
        cp = pickle.load(cp_file)
    solver = cp['solver']

    # Set attributes to solver
    # solver.geometry.src_positions=src_positions
    geometry = AcquisitionGeometry(true_model, rec_coordinates, src_positions,
                                param['t0'], param['tn'], src_type='Ricker',
                                f0=param['f0'])
    # Compute simulated data and full forward wavefield u0
    solver = AcousticWaveSolver(true_model, geometry, space_order=4)
    
    # solver.geometry.resample(dt)
    # d, u0 = solver.forward(vp=model0.vp,save=True)[0:2]
    # d, u0 = solver.forward(vp=model0.vp, dt=dt)[0:2]
    d, u0 = solver.forward(vp=true_model.vp, dt=dt, save=True)[0:2]
    # d, u0 = solver.forward(vp=model0.vp, dt=dt)[0:2]
    print()