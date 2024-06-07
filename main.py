import numpy as onp
import jax.numpy as np
from erg_expl import ErgodicTrajectoryOpt
from IPython.display import clear_output
from gaussian import gaussian, gaussian_measurement
import matplotlib.pyplot as plt
import time

size = 100
rand_gaussian1 = gaussian(size, 20, 35, 5)
rand_gaussian2 = gaussian(size, 85, 15, 10)
rand_gaussian3 = gaussian(size, 45, 25, 10)
rand_gaussian4 = gaussian(size, 75, 75, 10)
#meas_reduction = gaussian_measurement(size, 200, 180, 0)
pmap = rand_gaussian1 + rand_gaussian2 + rand_gaussian3 + rand_gaussian4
pmap[pmap<0]= 1E-10
num_agents = 2
init_pos = np.array([[12., 50.], [30., 50.]])
bounds = np.array([[0.,100.],[0.,100.]])

traj_opt = ErgodicTrajectoryOpt(init_pos, pmap, num_agents, bounds)
for k in range(100):
    traj_opt.solver.solve(max_iter=100000)
    sol = traj_opt.solver.get_solution()
    clear_output(wait=True)

fig, ax = plt.subplots()
ax.imshow(pmap.T, origin="lower")
ax.plot(sol['x'][:,0][:,0], sol['x'][:,0][:,1], color='r')
ax.plot(sol['x'][:,1][:,0], sol['x'][:,1][:,1], color='b')
plt.show()

'''
fig = plt.figure()
plt.imshow(map.T, origin="lower")
plt.show()
'''
