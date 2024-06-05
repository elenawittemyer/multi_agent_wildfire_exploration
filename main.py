import numpy
import jax.numpy as np
from erg_expl import ErgodicTrajectoryOpt

initpos=[0,0]
initmap=numpy.zeros((100,100))
traj_opt = ErgodicTrajectoryOpt(initpos, initmap, "t")