from jax import vmap
import jax.numpy as np
import jax.scipy as jsp
import numpy as onp
import matplotlib.pyplot as plt
from gaussian import gaussian_1d, gaussian, gaussian_filter
from data_and_plotting.fluid_engine_dev.src.examples.python_examples.smoke_example01 import gen_smoke
import os
import cv2
import time

## Collect new data: gen_smoke(log_data=True, grid_size=100)

def vis_array(frame, size, cells):
    den_cutoff = .3
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

def vis_array_b(frame, size, cells, peak_indices):
    den_cutoff = .3

    peaks = np.empty((0, 2))
    change_idx = []
    change_sum = 0
    for peak in peak_indices: 
        peak = np.flip(np.array(peak))
        peaks = np.vstack((peaks, peak.T))
        change_sum += len(peak.T)
        change_idx.append(change_sum)
    change_idx = np.array(change_idx)

    den = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy')
    den_avg = []
    for i in range(len(peak_indices)):
        den_avg.append(np.average(den[peak_indices[i]]))
    den_avg = np.array(den_avg)

    den_array = []
    for cell in cells:
        den_array.append(den[int(cell[0]), int(cell[1])])
    den_array = np.array(den_array)

    vis_array = []
    for i in range(len(cells)):
        if cells[i].tolist() in peaks.tolist():
            j = np.where(np.equal(peaks, cells[i]).all(1))[0][0]
            k = np.where(j<change_idx)[0][0]
            vis_array.append(max((den_cutoff-den_avg[k])/den_cutoff, 0))
        else:
            vis_array.append(max((den_cutoff-den_array[i])/den_cutoff, 0))
    
    return np.array(vis_array)

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
    avoid_smoke = True
    den_cutoff = .3

    # load smoke density grid and calculate visibility at measurement location
    den = np.abs(np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy'))
    
    if avoid_smoke == True:
        inv_den = np.maximum(den_cutoff-den, np.zeros(len(den)))
        vis = 1-inv_den/np.max(inv_den)
    else:
        vis = 1-den/np.max(den)
    
    C = vis[x[0], x[1]]

    # apply gaussian filter
    gauss_prob = gaussian_1d(10, 4.5, 4.5)
    gauss_prob /= np.sum(gauss_prob)
    v_noise_avg = np.sum(np.multiply(V, gauss_prob))
    V_noise_adj = v_noise_avg*np.ones(10)

    # create gaussian peak centered at visibility coefficient
    spread = 100/v_noise_avg
    rad = spread/2 # smoke vs peak effect seems balanced?
    pdf_V = []
    gauss_peak = gaussian_1d(200, C*100, rad)
    
    # find probability of observing each measurement in possible measurement values
    for v in V_noise_adj:
        pdf_V.append(gauss_peak[(np.int32(v*spread))])
    pdf_V = np.array(pdf_V)

    return np.array(pdf_V)

def _shannon_entropy(V, x, map_args):
    # find pdf of possible measurement values
    pdf_array = pdf(V, x, map_args)
    return -1*np.sum(pdf_array * np.log(pdf_array)/np.log(2))

shannon_entropy = vmap(_shannon_entropy, in_axes=(0, 0, None))

def calc_entropy(map, size, frame, noise_on = True):
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

    # calculate possible measurement values, adding in noise
    def measure_noise(v, noise_on):
        if noise_on == True:
            return onp.random.normal(v, .2, 10)
        else:
            return onp.random.normal(v, 0, 10)

    V_array = []
    for v in v_array:
        V_array.append(np.sort(measure_noise(v, noise_on)))
    V_array = np.array(V_array)

    # calculate shannon entropy at all coordinates
    info_grid = shannon_entropy(V_array, out, args)
    info_grid = info_grid.reshape((args['size'], args['size']))

    '''
    denoised_info = cv2.fastNlMeansDenoising(onp.uint8(onp.copy(info_grid)), None, h=.1, templateWindowSize=7, searchWindowSize=21)   
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax2.imshow(info_grid, origin='lower')
    ax1.imshow(map, origin='lower')
    den = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy')
    ax1.imshow(den, vmin=0, vmax=1, alpha = .5, cmap=plt.cm.gray, interpolation='nearest', origin='lower')
    ax3.imshow(denoised_info, origin='lower')
    plt.show()
    '''
    
    return info_grid

def calc_mask_map(map, size, frame, noise_on = True):
    den_cutoff = .3

    X,Y = onp.meshgrid(onp.arange(size), onp.arange(size))
    out = onp.column_stack((Y.ravel(), X.ravel()))

    # find info measurement at all coordinates
    def _eval_map(x, map):
        return map[x[0], x[1]]
    eval_map = vmap(_eval_map, in_axes=(0, None))

    v_array = eval_map(out, map)

    # apply noise to measurement
    def measure_noise(v):
        if noise_on == True:
            return np.abs(onp.random.normal(v, .2))
        else:
            return np.abs(onp.random.normal(v, 0))
    v_noise_array = []
    for v in v_array:
        v_noise_array.append(measure_noise(v))
    v_noise_array = np.array(v_noise_array)
    noisy_map = np.reshape(v_noise_array, (size, size))
    
    # apply visibility coefficients to noisy map
    den = np.abs(np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy'))
    den_norm = den/np.max(den)
    vis_red = np.maximum((den_cutoff-den_norm)/den_cutoff, np.zeros(den.shape))
    mask_map = np.multiply(vis_red, noisy_map)

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
    den_cutoff = .3
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

'''
size = 150
total_smoke = []
for i in range(150):
    den = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(i) + '.npy')
    total_smoke.append(np.sum(den))
plt.plot(range(len(total_smoke)), total_smoke)
plt.show()

#WARNING: ONLY USE ONCE
def modify_smoke(size):
    _, _, files = next(os.walk("data_and_plotting/smoke_density/smoke_grid_" + str(size)))
    file_count = len(files)
    for i in range(file_count):
        init_den = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(i) + '.npy')
        mod_den = init_den*(1+i/400)        
        with open('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(i) + '.npy', 'wb') as f:
            np.save(f, mod_den)
    return 0
'''
