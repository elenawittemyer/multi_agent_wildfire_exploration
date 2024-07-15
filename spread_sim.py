import jax.numpy as np
from jax import vmap
import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

#TODO: something is wrong with quenching, active areas are not always correct

def adj_fill(cell, map, zero_indices):
    size = map.shape[0]
    adj_cells = np.array([cell+np.array([0, 1]), cell+np.array([1, 0]), cell+np.array([1, 1]), cell-np.array([0, 1]), 
                          cell-np.array([1, 0]), cell-np.array([1, 1]), cell+np.array([1, -1]), cell+np.array([-1, 1])])
    adj_cells = np.clip(adj_cells, 0, size-1)
    adj_cells = tuple((adj_cells[:,0], adj_cells[:,1]))
    map = map.at[adj_cells].set(2)
    map = map.at[zero_indices].set(0)
    return map

def adj_quench(cell, map, zero_indices):
    size = map.shape[0]
    adj_cells = np.array([cell+np.array([0, 1]), cell+np.array([1, 0]), cell+np.array([1, 1]), cell-np.array([0, 1]), 
                          cell-np.array([1, 0]), cell-np.array([1, 1]), cell+np.array([1, -1]), cell+np.array([-1, 1])])
    adj_cells = np.clip(adj_cells, 0, size-1)
    adj_cells = tuple((adj_cells[:,0], adj_cells[:,1]))
    map = map.at[adj_cells].set(1)
    map = map.at[zero_indices].set(0)
    return map

def init_sim(map_size):
    fire_grid = np.ones((map_size, map_size))
    init_source = onp.random.randint(0, map_size, 2)
    fire_grid = fire_grid.at[init_source[0], init_source[1]].set(0)
    fire_grid = adj_fill(init_source, fire_grid, tuple((np.array([init_source[0], init_source[1]]))))
    return fire_grid, init_source

def step_fire(current_grid, source_array):
    # create new fire in danger area
    danger_area = np.where(current_grid == 2)
    ignite_idx = onp.random.randint(len(danger_area[0]))
    current_grid = current_grid.at[danger_area[0][ignite_idx], danger_area[1][ignite_idx]].set(0)

    # update danger area
    active_idx = np.where(current_grid == 0)
    source_array = source_array.tolist()
    source_array.append([danger_area[0][ignite_idx], danger_area[1][ignite_idx]])
    source_array = np.array(source_array)

    for source in source_array:
        current_grid = adj_fill(source, current_grid, active_idx)

    return current_grid, source_array

def main(map_size, t_f):
    fire_grid, init_source = init_sim(map_size)
    source_array = [init_source]

    img_data = []
    for i in range(1, t_f):
        if i>=5:
            fire_grid = fire_grid.at[source_array[0][0], source_array[0][1]].set(1)
            active_idx = np.where(fire_grid==0)
            fire_grid = adj_quench(source_array[0], fire_grid, active_idx)
            source_array = source_array.tolist()
            source_array.pop(0)
        
        img_data.append(fire_grid)
        fire_grid, source_array = step_fire(fire_grid, np.array(source_array))
    return img_data

def animate(data, t_f):
    fig, ax = plt.subplots()
    img = ax.imshow(data[0], animated=True)
    def updatefig(frame, img, ax):
        img.set_data(data[frame])
        return img,
    ani = animation.FuncAnimation(fig, updatefig, frames=t_f-1, fargs=(img, ax), interval=1, blit=True)
    mywriter = animation.FFMpegWriter(fps = 10)
    ani.save("data_and_plotting/videos/fire_spread.mp4", writer=mywriter)
    plt.close(fig)

data = main(100, 30)
animate(data, 30)