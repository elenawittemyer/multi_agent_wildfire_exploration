import jax.numpy as np
from jax import vmap
import numpy as onp
from gaussian import gaussian, gaussian_measurement
from data_and_plotting.plotting import final_plot, get_colormap, plot_info_reduct
from data_and_plotting.fluid_engine_dev.src.examples.python_examples.smoke_example01 import gen_smoke
import matplotlib.pyplot as plt
import time
import os

from moving_targets import dynamic_info_init, dynamic_info_step
from smoke import vis_array, vis_array_b

def greedy_step(info_map, starts, N, t_u, max_step):
    flat_map = info_map.flatten()
    sort_map = np.argsort(flat_map)
    max_idxs_1d = sort_map[len(sort_map):len(sort_map)-N-1:-1]
    max_idxs = np.unravel_index(max_idxs_1d, info_map.shape)
    
    multi_traj = []
    for i in range(N):
        start_pos = starts[i]
        end_pos = np.array([max_idxs[1][i], max_idxs[0][i]]) 

        dist = np.linalg.norm(end_pos-start_pos)
        angle = np.arctan2(end_pos[1]-start_pos[1], end_pos[0]-start_pos[0])
        
        num_max_step = int(dist//max_step)

        traj = []
        for j in range(num_max_step):
            traj.append(np.array([start_pos[0]+j*max_step*np.cos(angle), start_pos[1]+j*max_step*np.sin(angle)]))
        if num_max_step==0:
            traj.append(start_pos)
        
        num_zero_step = t_u-len(traj)
        for j in range(num_zero_step):
            traj.append(end_pos)

        traj = np.array(traj)
        multi_traj.append(traj)
    multi_traj = np.array(multi_traj)
    return multi_traj

def lawnmower_step(info_map, starts, N, t_u, step=1):
    starts = np.floor(starts).astype(int)

    multi_traj = []
    for i in range(N):
        flat_idx = np.ravel_multi_index((starts[i][0], starts[i][1]), info_map.shape)
        flat_path = np.arange(flat_idx, flat_idx+step*t_u, step)
        path_indices = np.unravel_index(flat_path, info_map.shape)
        traj = np.hstack((np.array([path_indices[0]]).T, np.array([path_indices[1]]).T))
        multi_traj.append(traj)
    multi_traj = np.array(multi_traj)
    return multi_traj

def baseline_main(t_f, t_u, peaks, num_agents, size, map_params, init_pos = None, dynamic_info = False, method='greedy'):

    init_map = map_params['init_map']
    peak_pos = map_params['peak_pos']
    target_pos = map_params['target_pos']
    target_vel = map_params['target_vel']

    if init_pos is None:
        init_pos = sample_initpos(num_agents, size)
    if init_map is None:
        if dynamic_info == True:
            init_map, peak_pos, target_pos, target_vel = dynamic_info_init(size, peaks)
            init_map = noise_mask(init_map)
        else:
            init_map, peak_pos = sample_map(size, peaks)
            init_map = noise_mask(init_map)
    cmap = get_colormap(num_agents+1)

    plot_prog = False
    record_info_red = True

    if os.path.isdir('data_and_plotting/smoke_density/smoke_grid_' + str(size)) == False:
        print('Generating smoke data... ')
        os.mkdir('data_and_plotting/smoke_density/smoke_grid_' + str(size))
        gen_smoke(log_data=True, grid_size=size)
    
    pmap = init_map
    erg_file = open('data_and_plotting/plotting_data/erg_metric_data.txt', 'w+')
    path_travelled = np.empty(shape=(num_agents, 2) + (0, )).tolist()
    map_sum = []
    init_pos = init_pos + np.array([size/2, size/2])

    for step in range(0, t_f, t_u):
        print(str(step/t_f*100) + "% complete")

        with open('data_and_plotting/dynamic_info_data/info_map_' + str(step//t_u) + '.npy', 'wb') as f:
            np.save(f, pmap)
        
        new_initpos = []
        for i in range(num_agents):
            if record_info_red == True:
                map_sum.append(np.sum(pmap))

            if method == 'greedy':
                sol = greedy_step(pmap, init_pos, num_agents, t_u, 10)
            elif method == 'lawnmower':
                sol = lawnmower_step(pmap, init_pos, num_agents, t_u, 10)
            else:
                print('unknown method')
                break

            path_travelled[i][0].append(sol[i][:,0])
            path_travelled[i][1].append(sol[i][:,1])
            pmap = update_map(np.floor(sol[i]), pmap, step, size, peak_pos)                        
            new_initpos.append(sol[i][-1])
            
        if plot_prog == True:
            smoke_grid = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(step) + '.npy')
            fig, ax1 = plt.subplots()
            ax1.imshow(pmap, origin='lower')
            for i in range(num_agents):
                ax1.plot(np.array(path_travelled[i][0]).flatten(), np.array(path_travelled[i][1]).flatten(), c=cmap(i))
            ax1.imshow(smoke_grid, vmin=0, vmax=1, alpha = .5, cmap=plt.cm.gray, interpolation='nearest', origin='lower')
            plt.show()
        
        if record_info_red == True:
                map_sum.append(np.sum(pmap))
        init_pos = np.array(new_initpos)

        if dynamic_info == True:
            pmap, peak_pos, target_pos, target_vel = dynamic_info_step(peaks, size, pmap, peak_pos, target_pos, target_vel)
    
    erg_file.close()

    if record_info_red == True:
        map_file = open('data_and_plotting/plotting_data/info_map_data.txt', 'w+')
        for val in map_sum:
            map_file.write(str(val) + ' \n')
        map_file.close()
    
    return path_travelled, init_map, pmap


################################
## Mapping Helpers #############
################################

def normalize_map(map):
    return map / np.sum(np.abs(map))

def _measure_update(cell, size):
    reduction = np.array(gaussian_measurement(size, cell[0], cell[1], .03))
    return reduction
measure_update = vmap(_measure_update, in_axes=(0, None))

def update_map(current_pos, current_map, iter, size, peak_idx):
    blackout_info = True
    smoke_on = True

    if smoke_on== True:
        if blackout_info == True:
            vis_coeffs = vis_array_b(iter, size, current_pos, peak_idx)
        else:
            vis_coeffs = vis_array(iter, size, current_pos)
    else:
        vis_coeffs = np.ones(len(current_pos))
    
    all_reductions = measure_update(current_pos, size)
    weighted_reductions = weight_mult(all_reductions, vis_coeffs)
    new_map = current_map + np.sum(weighted_reductions, axis=0)
    new_map = np.maximum(new_map, np.zeros(new_map.shape)+1E-10)
    return new_map

def _weight_mult(reduct_map, coeff):
    return coeff*reduct_map
weight_mult = vmap(_weight_mult, in_axes=(0, 0))

################################
## Sample Data #################
################################

def sample_map(size, num_peaks):
    pos = np.floor(onp.random.uniform(0, size, 2*num_peaks))
    pmap = gaussian(size, pos[0], pos[1], 10)
    peak_indices = [np.where(pmap>.1)]
    for i in range(1, num_peaks):
        new_peak = gaussian(size, pos[2*i], pos[2*i+1], 10)
        pmap += new_peak
        peak_indices.append(np.where(new_peak>.1))
    return pmap, peak_indices

def sample_initpos(num_agents, size):
    return onp.random.uniform(0, size, (num_agents, 2))

def noise_mask(map):
    noise = onp.random.uniform(1E-3, 1E-2, map.shape)
    noise = np.reshape(noise, map.shape)
    return map+noise

################################
## Testing #####################
################################
