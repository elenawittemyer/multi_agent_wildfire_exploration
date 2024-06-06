import numpy as onp
import jax.numpy as np
from erg_expl import ErgodicTrajectoryOpt
from IPython.display import clear_output
from gaussian import gaussian, gaussian_measurement
import matplotlib.pyplot as plt
import time

size = 500
rand_gaussian = gaussian(size, 200, 180, 130)
meas_reduction = gaussian_measurement(size, 200, 180, .5)
map = rand_gaussian + meas_reduction
map[map<0]=0
fig = plt.figure()
plt.imshow(map.T, origin="lower")
plt.show()

