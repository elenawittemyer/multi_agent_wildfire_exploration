from jax import vmap
import jax.numpy as np
import matplotlib.pyplot as plt
from .fluid_engine_dev.src.examples.python_examples.smoke_example01 import gen_smoke
import os
import time

## Collect new data: gen_smoke(log_data=True, grid_size=100)

def vis_array(frame, size, cells):
    den_cutoff = .25

    path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    smoke_grid = np.load(path + '/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy')
    den_array = []
    for cell in cells:
        den_array.append(smoke_grid[int(cell[0]), int(cell[1])])
    den_array = np.array(den_array)

    vis_array = []
    for val in den_array:
        if val<den_cutoff:
            vis_array.append(1-np.abs(val))
        else:
            vis_array.append(0)
    vis_array = np.array(vis_array)
    return vis_array

def safety_cost(frame, size, x):
    cells = np.floor(x)
    path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    smoke_grid = np.load(path + '/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy')
    danger_cells = np.where(smoke_grid>=.9)
    danger_grid = np.hstack((danger_cells[0][:, np.newaxis], danger_cells[1][:, np.newaxis]))
    cost = 0
    for cell in cells:
        pos_check = cell == danger_grid
        cost += np.sum(-1*np.multiply(pos_check[:,0], pos_check[:,1]))
    return cost

def safety_map(map, frame, size):
    path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    smoke_grid = np.load(path + '/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy')
    danger_cells = np.where(smoke_grid>=.9)
    danger_grid = np.hstack((danger_cells[0][:, np.newaxis], danger_cells[1][:, np.newaxis]))
    for cell in danger_grid:
        map.at[cell[0], cell[1]].set(0)
    return map