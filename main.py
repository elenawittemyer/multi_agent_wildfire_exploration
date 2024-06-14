import numpy as onp
import jax.numpy as np
from jax import vmap
from erg_expl import ErgodicTrajectoryOpt
from IPython.display import clear_output
from gaussian import gaussian, gaussian_measurement
import matplotlib.pyplot as plt
import time

#TODO: figure out how to force steps to be of a certain size

def main(t_f, t_u, num_agents): # final time, measurement frequency, agents. update time should be at most half of time horizon (since init_pos=end_pos).
    init_pos = sample_initpos(num_agents)
    pmap = sample_map(100)
    plot_prog = True
    cmap = get_colormap(num_agents+1)

    path_travelled = np.empty(shape=(num_agents, 2) + (0, )).tolist()
    for step in range(0, t_f, t_u):
        print(str(step/t_f*100) + "% complete")

        traj_opt = ErgodicTrajectoryOpt(np.floor(init_pos), pmap, num_agents)
        for k in range(100):
            traj_opt.solver.solve(max_iter=1000)
            sol = traj_opt.solver.get_solution()
            clear_output(wait=True)

        new_initpos = []
        for i in range(num_agents):
            path_travelled[i][0].append(sol['x'][:,i][:,0][:t_u]+50.)
            path_travelled[i][1].append(sol['x'][:,i][:,1][:t_u]+50.)
            pmap = update_map(np.array([sol['x'][:,i][:,0][:t_u], sol['x'][:,i][:,1][:t_u]]).T, pmap)                        
            new_initpos.append([sol['x'][:,i][:,0][t_u-1], sol['x'][:,i][:,1][t_u-1]])
            
            if plot_prog == True:
                fig, ax = plt.subplots()
                ax.imshow(pmap, origin="lower")
                ax.plot(np.array(path_travelled[i][0]).flatten(), np.array(path_travelled[i][1]).flatten(), c=cmap(i))
                plt.show()
        
        init_pos = np.array(new_initpos)

    return path_travelled, pmap
    
################################
## Mapping Helpers #############
################################

def normalize_map(map):
    return map / np.sum(np.abs(map))

def _measure_update(current_pos):
    cell = np.floor(current_pos)
    c = sample_vis_coeff()
    reduction = c*np.array(gaussian_measurement(100, cell[0]+50, cell[1]+50, .05))
    return reduction
measure_update = vmap(_measure_update, in_axes=(0))

def update_map(current_pos, current_map):
    all_reductions = measure_update(current_pos)
    new_map = current_map + np.sum(all_reductions, axis=0)
    new_map = np.maximum(new_map, np.zeros(new_map.shape)+1E-10)
    return new_map

################################
## Sample Data #################
################################

def sample_map(size):
    rand_gaussian1 = gaussian(size, 20, 35, 10)
    rand_gaussian2 = gaussian(size, 85, 15, 10)
    rand_gaussian3 = gaussian(size, 45, 25, 10)
    rand_gaussian4 = gaussian(size, 75, 75, 10)
    rand_gaussian5 = gaussian(size, 15, 95, 10)
    rand_gaussian6 = gaussian(size, 30, 30, 10)
    rand_gaussian7 = gaussian(size, 15, 5, 10)
    rand_gaussian8 = gaussian(size, 55, 85, 10)

    pmap = rand_gaussian1 + rand_gaussian2 + rand_gaussian3 + rand_gaussian4 + \
           rand_gaussian5 + rand_gaussian6 + rand_gaussian7 + rand_gaussian8
    
    return pmap

def sample_initpos(num_agents):
    return onp.random.uniform(-50, 50, (num_agents, 2))

def sample_vis_coeff():
    return .5

################################
## Plotting ####################
################################
     
def get_colormap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

################################
## Testing #####################
################################

agents = 2
path, map = main(100, 20, agents)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(map, origin="lower")
cmap = get_colormap(agents+1)

starts = []
for i in range(agents):
    ax1.plot(np.array(path[i][0]).flatten(), np.array(path[i][1]).flatten(), c=cmap(i))
    starts.append(plt.Circle(((np.array(path[i][0]).flatten()[0], np.array(path[i][1]).flatten()[0])), .3, color='w'))

for i in range(agents):
    ax1.add_patch(starts[i])

ax2.imshow(sample_map(100), origin="lower")
plt.show()