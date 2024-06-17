from jax import vmap
import jax.numpy as np
import matplotlib.pyplot as plt
from fluid_engine_dev.src.examples.python_examples.smoke_example01 import main
import time

## Collect new data: main(log_data=True)

def vis_array(frame, cells):
    smoke_grid = np.load('smoke_density/smoke_array_' + str(frame) + '.npy')
    den_array = []
    for cell in cells:
        den_array.append(smoke_grid[int(cell[0]), int(cell[1])])
    den_array = np.array(den_array)
    vis_array = np.maximum(1 - den_array, np.zeros(len(den_array)))
    return vis_array
