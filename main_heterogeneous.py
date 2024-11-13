import os
import numpy as onp
import jax.numpy as np
from jax import vmap
from baseline_methods import baseline_main
from erg_expl import SingleErgodicTrajectoryOpt, MultiErgodicTrajectoryOpt
from IPython.display import clear_output
from gaussian import gaussian, gaussian_measurement
from data_and_plotting.plotting import animate_dynamic_info, animate_targets, animate_vis, basic_path_plot, freq_plot, get_colormap, animate_plot, final_plot, plot_ergodic_metric, plot_info_reduct
from moving_targets import dynamic_info_init, dynamic_info_step
from smoke import vis_array, calc_entropy, calc_mask_map, blackout_map, vis_array_b
from data_and_plotting.fluid_engine_dev.src.examples.python_examples.smoke_example01 import gen_smoke
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
import time
import math

def main(t_f, t_u, peaks, num_agents, size, map_params, init_pos = None, entropy=False, mask_map = False, blackout = False, dynamic_info = False, noise = True, motion_model = None, factor = 'speed'):
    init_map = map_params['init_map']
    peak_pos = map_params['peak_pos']
    target_pos = map_params['target_pos']
    target_vel = map_params['target_vel']

    if init_pos is None:
        init_pos = sample_initpos(num_agents, size)
    
    if init_map is None:
        if dynamic_info == True:
            init_map, peak_pos, target_pos, target_vel = dynamic_info_init(size, peaks)
        else:
            init_map, peak_pos = sample_map(size, peaks)
  
    if factor == 'speed':
        if motion_model is None:
            motion_model = sample_motion_model(num_agents)
        cam_coeff = .3*np.ones(num_agents)
    elif factor == 'camera':
        cam_coeff = sample_vis_coeffs(num_agents)
        motion_model = 10.0*np.ones(num_agents)
   
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

    for step in range(0, t_f, t_u):
        if factor == 'speed':
            maps = spectral_decomp(pmap, size, num_agents)
        
        print(str(step/t_f*100) + "% complete")
        
        with open('data_and_plotting/dynamic_info_data/info_map_' + str(step//t_u) + '.npy', 'wb') as f:
            np.save(f, pmap)

        new_initpos = []
        for agent in range(num_agents):
            
            if factor == 'speed':
                pmap_agent = maps[agent]
            elif factor == 'camera':
                pmap_agent = np.copy(pmap)

            if entropy == True:
                opt_map = calc_entropy(pmap_agent, size, step, noise, cam_coeff[agent])
            elif mask_map == True:
                opt_map = calc_mask_map(pmap_agent, size, step, noise, cam_coeff[agent])
            elif blackout == True:
                opt_map = blackout_map(pmap_agent, peak_pos, size, step, cam_coeff[agent])
            else:
                opt_map = apply_meas_noise(pmap_agent, size, noise)

            traj_opt = SingleErgodicTrajectoryOpt(np.floor(init_pos[agent]), opt_map, size, erg_file, motion_model[agent])
            for k in range(100):
                traj_opt.solver.solve(max_iter=1000)
                sol = traj_opt.solver.get_solution()
                clear_output(wait=True)

            if record_info_red == True:
                map_sum.append(np.sum(pmap))

            path_travelled[agent][0].append(sol['x'][:,0][:t_u]+(size/2))
            path_travelled[agent][1].append(sol['x'][:,1][:t_u]+(size/2))
            
            pmap = update_map(np.floor(np.array([sol['x'][:,0][:t_u], sol['x'][:,1][:t_u]]).T)+(size/2), pmap, step, size, peak_pos, cam_coeff[agent])

            new_initpos.append([sol['x'][:,0][t_u-1], sol['x'][:,1][t_u-1]])
            
        if plot_prog == True:
            smoke_grid = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(step) + '.npy')
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(pmap, origin="lower")
            for i in range(num_agents):
                x_traj = np.array(path_travelled[i][0]).flatten()
                y_traj = np.array(path_travelled[i][1]).flatten()

                if step==0:
                    ax1.plot(x_traj, y_traj, c=cmap(i))
                    ax2.plot(x_traj, y_traj, c=cmap(i))
                else:
                    ax1.plot(x_traj[0:len(x_traj)-20], y_traj[0:len(x_traj)-20], c=cmap(i), alpha=0.3)
                    ax1.plot(x_traj[len(x_traj)-20:], y_traj[len(x_traj)-20:], c=cmap(i))
                    ax2.plot(x_traj[0:len(x_traj)-20], y_traj[0:len(x_traj)-20], c=cmap(i), alpha=0.3)
                    ax2.plot(x_traj[len(x_traj)-20:], y_traj[len(x_traj)-20:], c=cmap(i))
            
            ax1.imshow(smoke_grid, vmin=0, vmax=1, alpha = .5, cmap=plt.cm.gray, interpolation='nearest', origin='lower')
            ax2.imshow(opt_map, origin='lower')
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
## Map Decomposition ###########
################################
def grid_splitter(map, map_size, agents):
    points = onp.random.uniform(0, map_size, (500, 2)).reshape(2,500)
    data = list(zip(points[0], points[1]))
    kmeans = KMeans(n_clusters=agents)
    kmeans.fit(data)
    centroids  = kmeans.cluster_centers_ 

    x, y = np.meshgrid(np.arange(map_size), np.arange(map_size))
    grid_points = np.column_stack((x.ravel(), y.ravel()))
    tree = cKDTree(centroids)
    distances, indices = tree.query(grid_points)
    vor_region_matrix = indices.reshape((map_size, map_size))
    #plt.imshow(vor_region_matrix, cmap='nipy_spectral', origin='lower')
    #plt.show()

    maps = []
    for i in range(agents):
        agent_map = np.zeros((map_size, map_size))
        nonzero_idx = np.where(vor_region_matrix==i)
        agent_map = agent_map.at[nonzero_idx].set(map[nonzero_idx])
        maps.append(agent_map)
        
    return maps

################################
## Mapping Helpers #############
################################

def normalize_map(map):
    return map / np.sum(np.abs(map))

def _measure_update(cell, size):
    reduction = np.array(gaussian_measurement(size, cell[0], cell[1], .03))
    return reduction
measure_update = vmap(_measure_update, in_axes=(0, None))

def update_map(current_pos, current_map, iter, size, peak_idx, den_cutoff = .3):
    blackout_info = True
    smoke_on = True

    if smoke_on== True:
        if blackout_info == True:
            vis_coeffs = vis_array_b(iter, size, current_pos, peak_idx, den_cutoff)
        else:
            vis_coeffs = vis_array(iter, size, current_pos, den_cutoff)
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

def spectral_decomp(_info_map, size, num_agents, plot_freq = False):
    map_decomp = np.fft.fft2(_info_map)
    map_decomp = np.fft.fftshift(map_decomp)
    bands = map_decomp.real

    if plot_freq == True:
        fft_x = np.fft.fftfreq(bands.shape[0])
        fft_y = np.fft.fftfreq(bands.shape[1])
        fft_freqs = []
        fft_weights = []
        
        freq_index_map = {}
        freq_map = onp.zeros((size, size))
        for i in range(bands.shape[0]):
            for j in range(bands.shape[1]):
                freq = float(np.sqrt(fft_x[i]**2 + fft_y[j]**2))
                freq_map[i][j] = freq
                
                if freq in freq_index_map:
                    idx = freq_index_map[freq]
                    fft_weights[idx] += bands[i][j]
                else:
                    idx = len(fft_freqs)
                    fft_freqs.append(freq)
                    fft_weights.append(bands[i][j])
                    freq_index_map[freq] = idx

        
        fft_freqs = np.array(fft_freqs)
        fft_weights = np.array(fft_weights)
        fft_stack = np.stack((fft_freqs, fft_weights), axis=-1)
        fft_stack = fft_stack[np.argsort(fft_stack[:,0])]
        fft_freqs, fft_weights = np.hsplit(fft_stack, 2)
        
        plt.plot(fft_freqs, fft_weights)
        plt.show()
    
    maps = spectral_recomp(_info_map, map_decomp, num_agents)

    return maps

def spectral_recomp(info_map, fft_result, num_agents):
    
    rows, cols = fft_result.shape
    center_row, center_col = rows // 2, cols // 2

    rad = 2
    bands = []
    for i in range(num_agents):
        offset = math.floor(rad*i**.7)
        width = (i+1)*rad
        low_freq_array = onp.zeros_like(fft_result)

        low_freq_array[center_row - rad:center_row + width, center_col - rad:center_col + width] = \
            fft_result[center_row - rad:center_row + width, center_col - rad:center_col + width]
        if offset!=0:
            low_freq_array[center_row-offset:center_row+offset, center_col-offset:center_col+offset] = 0

        bands.append(low_freq_array)

    fig, axs = plt.subplots(2, 3)
    maps = []
    for i in range(num_agents):
        reconstructed_array = np.fft.ifft2(np.fft.ifftshift(bands[i])).real
        maps.append(reconstructed_array)
        ax = axs[i//3, i%3]
        ax.imshow(reconstructed_array, origin='lower')

    ax = axs[1, 2]
    ax.imshow(info_map, origin='lower')
    plt.show()

    return maps

################################
## Sample Data #################
################################

def sample_map(size, num_peaks):
    pos = np.floor(onp.random.uniform(5, size-5, 2*num_peaks))
    pmap = gaussian(size, pos[0], pos[1], 10)
    peak_indices = [np.where(pmap>.1)]
    for i in range(1, num_peaks):
        new_peak = gaussian(size, pos[2*i], pos[2*i+1], 10)
        pmap += new_peak
        peak_indices.append(np.where(new_peak>.1))
    return pmap, peak_indices

def sample_initpos(num_agents, size):
    #return np.zeros((num_agents, 2))
    return onp.random.uniform(-size/2, size/2, (num_agents, 2))

def sample_vis_coeffs(num_agents):
    return np.linspace(.1, .5, num_agents)

def sample_motion_model(num_agents):
    return np.flip(np.linspace(1, 20, num_agents))

def apply_meas_noise(map, size, noise_on):
    X,Y = onp.meshgrid(onp.arange(size), onp.arange(size))
    out = onp.column_stack((Y.ravel(), X.ravel()))

    # find info measurement at all coordinates
    def _eval_map(x, map):
        return map[x[0], x[1]]
    eval_map = vmap(_eval_map, in_axes=(0, None))

    v_array = eval_map(out, map)

    # calculate possible measurement values, adding in noise
    def measure_noise(v):
        if noise_on == True:
            return np.abs(onp.random.normal(v, .15))
        else:
            return np.abs(onp.random.normal(v, 0))
        
    v_noise_array = []
    for v in v_array:
        v_noise_array.append(measure_noise(v))
    v_noise_array = np.array(v_noise_array)
    noisy_map = np.reshape(v_noise_array, (size, size))
    
    return noisy_map

################################
## Testing #####################
################################

agents = 5
t_f = 40
t_u = 20
size = 100
peaks = 5

comp_map, comp_peaks = sample_map(size, peaks)
#comp_map, comp_peaks, comp_targets, comp_vel = dynamic_info_init(size, peaks)
comp_pos = sample_initpos(agents, size)

speeds = sample_motion_model(agents)
#speeds = 8*np.ones(agents)

map_params = {
    'init_map': comp_map,
    'peak_pos': comp_peaks,
    'target_pos': None, #comp_targets
    'target_vel': None #comp_vel
}

maps = spectral_decomp(comp_map, size, agents)

path, i_map, f_map = main(t_f, t_u, peaks, agents, size, map_params, init_pos = comp_pos, mask_map = True, noise = False, motion_model = speeds, factor = 'speed')
final_plot(path, i_map, i_map, agents, t_f, speeds, True)
freq_plot(path, maps, agents)
plot_ergodic_metric()
plot_info_reduct(t_f, t_u, agents)
