import numpy as onp
import jax.numpy as np
from erg_expl import ErgodicTrajectoryOpt
from IPython.display import clear_output
from gaussian import gaussian, gaussian_measurement
import matplotlib.pyplot as plt
import time
from itertools import cycle

#TODO: not working for more than 2 agents?

def normalize_map(map):
    map[map<0]= 1E-10
    return map / np.sum(map)

def sample_map(size):
    rand_gaussian1 = gaussian(size, 20, 35, 10)
    rand_gaussian2 = gaussian(size, 85, 15, 10)
    rand_gaussian3 = gaussian(size, 45, 25, 10)
    rand_gaussian4 = gaussian(size, 75, 75, 10)
    rand_gaussian5 = gaussian(size, 15, 95, 10)
    rand_gaussian6 = gaussian(size, 30, 30, 10)
    rand_gaussian7 = gaussian(size, 15, 5, 10)
    rand_gaussian8 = gaussian(size, 55, 85, 10)

    pmap = rand_gaussian1 + rand_gaussian2 + rand_gaussian3 + rand_gaussian4 +rand_gaussian5 + rand_gaussian6 + rand_gaussian7 + rand_gaussian8
    return normalize_map(pmap)

def main(num_agents, init_pos, pmap):
    traj_opt = ErgodicTrajectoryOpt(init_pos, pmap, num_agents)
    for k in range(100):
        traj_opt.solver.solve(max_iter=1000)
        sol = traj_opt.solver.get_solution()
        clear_output(wait=True)
    fig, ax = plt.subplots()
    ax.imshow(pmap, origin="lower")
    cmap = get_colormap(num_agents+1)
    for i in range(num_agents):
        ax.plot(sol['x'][:,i][:,0]+50., sol['x'][:,i][:,1]+50., color=cmap(i))
    plt.show()
    return sol
    
def get_colormap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

main(5, np.array([[10., 37.], [-15., 1.], [20. , -18.], [8, -17], [38, 23]]), sample_map(100))