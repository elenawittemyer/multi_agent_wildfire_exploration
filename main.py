import numpy as onp
import jax.numpy as np
from erg_expl import ErgodicTrajectoryOpt
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time

initpos=[0,0]
initmap=onp.ones((100,100))
traj_opt = ErgodicTrajectoryOpt(initpos, initmap)

xpoints=onp.linspace(-1,1,100)
ypoints=onp.linspace(-1,1,100)

X, Y = traj_opt.target_distr.domain
P = traj_opt.target_distr.evals[0].reshape(X.shape)

for _ in range(1000):
    traj_opt.solver.solve(max_iter=50)
    sol = traj_opt.solver.get_solution()
    clear_output(wait=True)

plt.contour(xpoints, ypoints, initmap)
plt.plot(sol['x'][:,0], sol['x'][:,1])

plt.axis('equal')
plt.show()
time.sleep(.1)
plt.close()

