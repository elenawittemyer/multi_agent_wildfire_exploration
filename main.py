import numpy as onp
import jax.numpy as np
from jax import vmap
from erg_expl import ErgodicTrajectoryOpt
from IPython.display import clear_output
from gaussian import gaussian, gaussian_measurement
from data_and_plotting.plotting import animate_vis, get_colormap, animate_plot, final_plot, smoke_vs_info, time_dstrb_comp, plot_ergodic_metric, plot_info_reduct
from smoke import vis_array, calc_entropy, calc_mask_map, blackout_map, vis_array_b
from data_and_plotting.fluid_engine_dev.src.examples.python_examples.smoke_example01 import gen_smoke
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os

def main(t_f, t_u, peaks, num_agents, size, init_map=None, peak_pos = None, init_pos=None, entropy=False, mask_map = False, blackout = False):
    
    if init_pos is None:
        init_pos = sample_initpos(num_agents, size)
    if init_map is None:
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

    for step in range(0, t_f, t_u):
        print(str(step/t_f*100) + "% complete")

        if entropy == True:
            opt_map = calc_entropy(pmap, size, step)
        elif mask_map == True:
            opt_map = calc_mask_map(pmap, size, step)
        elif blackout == True:
            opt_map = blackout_map(pmap, peak_pos, size, step)
        else:
            opt_map = pmap
        
        traj_opt = ErgodicTrajectoryOpt(np.floor(init_pos), opt_map, num_agents, size, erg_file)
        for k in range(100):
            traj_opt.solver.solve(max_iter=1000)
            sol = traj_opt.solver.get_solution()
            clear_output(wait=True)

        new_initpos = []
        for i in range(num_agents):
            if record_info_red == True:
                map_sum.append(np.sum(pmap))

            path_travelled[i][0].append(sol['x'][:,i][:,0][:t_u]+(size/2))
            path_travelled[i][1].append(sol['x'][:,i][:,1][:t_u]+(size/2))
            pmap = update_map(np.floor(np.array([sol['x'][:,i][:,0][:t_u], sol['x'][:,i][:,1][:t_u]]).T)+(size/2), pmap, step, size, peak_pos)                        
            new_initpos.append([sol['x'][:,i][:,0][t_u-1], sol['x'][:,i][:,1][t_u-1]])
            
            if plot_prog == True:
                smoke_grid = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(step) + '.npy')
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(pmap, origin="lower")
                ax1.plot(np.array(path_travelled[i][0]).flatten(), np.array(path_travelled[i][1]).flatten(), c=cmap(i))
                ax1.imshow(smoke_grid, vmin=0, vmax=1, alpha = .5, cmap=plt.cm.gray, interpolation='nearest', origin='lower')
                ax2.imshow(opt_map, origin='lower')
                plt.show()
        
        init_pos = np.array(new_initpos)
    
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
    reduction = np.array(gaussian_measurement(size, cell[0], cell[1], .04))
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
        pmap += gaussian(size, pos[2*i], pos[2*i+1], 10)
        peak_indices.append(np.where(new_peak>.1))
    return pmap, peak_indices

def sample_initpos(num_agents, size):
    return onp.random.uniform(-size/2, size/2, (num_agents, 2))

def sample_vis_coeff():
    return .5

def noise_mask(map):
    noise = onp.random.uniform(1E-3, 1E-2, map.shape)
    noise = np.reshape(noise, map.shape)
    return map+noise

################################
## Testing #####################
################################

agents = 2
t_f = 100
t_u = 20
size = 100
peaks = 6

comp_map, comp_peaks = sample_map(size, peaks)
comp_pos = sample_initpos(agents, size)

path, i_map, f_map = main(t_f, t_u, peaks, agents, size, init_map = comp_map, peak_pos = comp_peaks, init_pos = comp_pos, blackout = True)
#path_ns, i_map, f_map_ns = main(t_f, t_u, peaks, agents, size, smoke_state=False, init_map=comp_map, init_pos=comp_pos)

#time_dstrb_comp(size, t_f, i_map, path_ns, path, agents, f_map, f_map_ns)
#animate_vis(size, t_f, i_map, path, agents, comp_peaks)
final_plot(path, i_map, f_map, agents, t_f)
plot_ergodic_metric()
plot_info_reduct(t_f)



path, i_map, f_map = main(t_f, t_u, peaks, agents, size, init_map = comp_map, peak_pos = comp_peaks, init_pos = comp_pos)
final_plot(path, i_map, f_map, agents, t_f)
plot_ergodic_metric()
plot_info_reduct(t_f)
