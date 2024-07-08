from jax import vmap
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
from gaussian import gaussian_1d, gaussian
from data_and_plotting.fluid_engine_dev.src.examples.python_examples.smoke_example01 import gen_smoke
import os
import time

## Collect new data: gen_smoke(log_data=True, grid_size=100)

def vis_array(frame, size, cells):
    den_cutoff = .45
    smoke_grid = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy')
    smoke_grid = np.abs(smoke_grid / np.max(smoke_grid))
    
    den_array = []
    for cell in cells:
        den_array.append(smoke_grid[int(cell[0]), int(cell[1])])
    den_array = np.array(den_array)

    vis_array = []
    for val in den_array:
        if val<den_cutoff:
            vis_array.append((den_cutoff-val)/den_cutoff)
        else:
            vis_array.append(0)
    vis_array = np.array(vis_array)

    return vis_array

def safety_cost(frame, size, x):
    cells = np.floor(x)
    smoke_grid = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy')
    danger_cells = np.where(smoke_grid>=.9)
    danger_grid = np.hstack((danger_cells[0][:, np.newaxis], danger_cells[1][:, np.newaxis]))
    cost = 0
    for cell in cells:
        pos_check = cell == danger_grid
        cost += np.sum(-1*np.multiply(pos_check[:,0], pos_check[:,1]))
    return cost

def pdf(V, x, args):
    frame = args['frame']
    size = args['size']
    avoid_smoke = False
    den_cutoff = .25

    # load smoke density grid and calculate visibility at measurement location
    den = np.abs(np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy'))
    
    if avoid_smoke == True:
        inv_den = np.maximum(den_cutoff-den, np.zeros(len(den)))
        vis = 1-inv_den/np.max(inv_den)
    else:
        vis = 1-den/np.max(den)
    
    C = vis[x[0], x[1]]
    
    # create gaussian peak centered at visibility coefficient
    v_true = V[int(len(V)/2)]
    spread =100/v_true
    rad = spread/2 # smoke vs peak effect seems balanced?
    pdf_V = []
    gauss_peak = gaussian_1d(200, C*100, rad)

    # find probability of observing each measurement in possible measurement values
    for v in V:
        pdf_V.append(gauss_peak[(np.int32(v*spread))])
    return np.array(pdf_V)

def _shannon_entropy(V, x, map_args):
    # find pdf of possible measurement values
    pdf_array = pdf(V, x, map_args)
    return -1*np.sum(pdf_array * np.log(pdf_array)/np.log(2))

shannon_entropy = vmap(_shannon_entropy, in_axes=(0, 0, None))

def calc_entropy(map, size, frame):
    args = {
    'frame': frame,
    'size': size
    }

    # create coordinate grid for exploration space
    X,Y = onp.meshgrid(onp.arange(args['size']), onp.arange(args['size']))
    out = onp.column_stack((Y.ravel(), X.ravel()))

    # find info measurement at all coordinates
    def _eval_map(x, map):
        return map[x[0], x[1]]
    eval_map = vmap(_eval_map, in_axes=(0, None))

    v_array = eval_map(out, map)

    # calculate possible measurement values, accounting for noise
    def _measure_noise(v):
        noise = v/5
        return np.linspace(v-noise, v+noise, 10)
    measure_noise = vmap(_measure_noise, in_axes=0)

    V_array = measure_noise(v_array)

    # calculate shannon entropy at all coordinates
    info_grid = shannon_entropy(V_array, out, args)
    info_grid = info_grid.reshape((args['size'], args['size']))

    '''
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(info_grid, origin='lower')
    ax2.imshow(map, origin='lower')
    den = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy')
    ax2.imshow(den, vmin=0, vmax=1, alpha = .5, cmap=plt.cm.gray, interpolation='nearest', origin='lower')
    plt.show()
    '''

    return info_grid

def calc_mask_map(map, size, frame):
    den_cutoff = .25
    
    den = np.abs(np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy'))
    den_norm = den/np.max(den)
    vis_red = np.maximum((den_cutoff-den_norm)/den_cutoff, np.zeros(den.shape))
    mask_map = np.multiply(vis_red, map)

    '''
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(mask_map, origin='lower')
    ax2.imshow(map, origin='lower')
    den = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy')
    ax2.imshow(den, vmin=0, vmax=1, alpha = .5, cmap=plt.cm.gray, interpolation='nearest', origin='lower')
    plt.show()
    '''

    return mask_map


def blackout_map(info_map, peak_indices, size, frame):
    den_cutoff = .45
    map = onp.copy(info_map)

    den = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy')
    den_avg = []
    for i in range(len(peak_indices)):
        den_avg.append(np.average(den[peak_indices[i]]))
    den_avg = np.array(den_avg)

    blackout_array = den_avg<den_cutoff 
    blackout_array *= 1

    for i in range(len(peak_indices)):
        map[peak_indices[i]] *= blackout_array[i]
    
    return np.copy(map)

size = 100
total_smoke = []
for i in range(100):
    den = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(i) + '.npy')
    total_smoke.append(np.sum(den))

'''
#WARNING: ONLY USE ONCE
def modify_smoke(size):
    _, _, files = next(os.walk("data_and_plotting/smoke_density/smoke_grid_" + str(size)))
    file_count = len(files)
    for i in range(file_count):
        init_den = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(i) + '.npy')
        mod_den = init_den*(1+i/450)        
        with open('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(i) + '.npy', 'wb') as f:
            np.save(f, mod_den)
    return 0
'''
