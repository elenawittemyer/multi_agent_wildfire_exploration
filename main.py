import numpy as onp
import jax.numpy as np
from jax import vmap
from erg_expl import ErgodicTrajectoryOpt
from IPython.display import clear_output
from gaussian import gaussian, gaussian_measurement
from plotting import get_colormap, animate_plot, final_plot, smoke_vs_info, time_dstrb_comp
from smoke import vis_array, safety_map
from fluid_engine_dev.src.examples.python_examples.smoke_example01 import gen_smoke
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os

#TODO: plot cumulative smoke density vs proportion of time in area to determine if algo is revisiting areas with poor visibility
#TODO: why do trajectories get squiggly for larger info maps?
    #TODO: write a function that punishes steps smaller than a certain size?


def main(t_f, t_u, peaks, num_agents, size, smoke_state=True):
    init_pos = sample_initpos(num_agents, size)
    init_map = sample_map(size, peaks)
    init_map = noise_mask(init_map)
    cmap = get_colormap(num_agents+1)

    plot_prog = False
    safety_aware = False

    if os.path.isdir('smoke_density/smoke_grid_' + str(size)) == False:
        print('Generating smoke data... ')
        os.mkdir('smoke_density/smoke_grid_' + str(size))
        gen_smoke(log_data=True, grid_size=size)

    pmap = init_map
    path_travelled = np.empty(shape=(num_agents, 2) + (0, )).tolist()
    for step in range(0, t_f, t_u):
        print(str(step/t_f*100) + "% complete")

        if safety_aware == True:
            opt_map = safety_map(pmap, step, size)
        else:
            opt_map = pmap
        
        traj_opt = ErgodicTrajectoryOpt(np.floor(init_pos), opt_map, num_agents, size)
        for k in range(100):
            traj_opt.solver.solve(max_iter=1000)
            sol = traj_opt.solver.get_solution()
            clear_output(wait=True)

        new_initpos = []
        for i in range(num_agents):
            path_travelled[i][0].append(sol['x'][:,i][:,0][:t_u]+(size/2))
            path_travelled[i][1].append(sol['x'][:,i][:,1][:t_u]+(size/2))
            pmap = update_map(np.floor(np.array([sol['x'][:,i][:,0][:t_u], sol['x'][:,i][:,1][:t_u]]).T)+(size/2), pmap, step, size, smoke_state)                        
            new_initpos.append([sol['x'][:,i][:,0][t_u-1], sol['x'][:,i][:,1][t_u-1]])
            
            if plot_prog == True:
                smoke_grid = np.load('smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(step) + '.npy')
                fig, ax = plt.subplots()
                ax.imshow(pmap, origin="lower")
                ax.plot(np.array(path_travelled[i][0]).flatten(), np.array(path_travelled[i][1]).flatten(), c=cmap(i))
                ax.imshow(smoke_grid, vmin=0, vmax=1, alpha = .5, cmap=plt.cm.gray, interpolation='nearest', origin='lower')
                plt.show()
        
        init_pos = np.array(new_initpos)

    return path_travelled, init_map, pmap
    
################################
## Mapping Helpers #############
################################

def normalize_map(map):
    return map / np.sum(np.abs(map))

def _measure_update(cell, size):
    reduction = np.array(gaussian_measurement(size, cell[0], cell[1], .05))
    return reduction
measure_update = vmap(_measure_update, in_axes=(0, None))

def update_map(current_pos, current_map, iter, size, smoke_state):
    if smoke_state == True:
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
    fixed = False
    if fixed==True:
        pos = np.array([10, 10, 20, 63, 90, 90, 55, 42])
        pmap = gaussian(size, pos[0], pos[1], 10)
        for i in range(1, 4):
            pmap += gaussian(size, pos[2*i], pos[2*i+1], 10)
    else:
        pos = np.floor(onp.random.uniform(0, size, 2*num_peaks))
        pmap = gaussian(size, pos[0], pos[1], 10)
        for i in range(1, num_peaks):
            pmap += gaussian(size, pos[2*i], pos[2*i+1], 10)
    return pmap

def sample_initpos(num_agents, size):
    fixed = False
    if fixed==True:
        return np.zeros((num_agents, 2))
    else:
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

agents = 3
t_f = 100
t_u = 20
size = 100
peaks = 8

path, init_map, fin_map = main(t_f, t_u, peaks, agents, size)

#path_ns, init_map, fin_map_ns = main(t_f, t_u, peaks, agents, size, smoke_state=False)
#time_dstrb_comp(size, t_f, init_map, path_ns, path_s, agents, fin_map_s, fin_map_ns)
#animate_plot(size, t_f, path)

final_plot(path, init_map, fin_map, agents, t_f)