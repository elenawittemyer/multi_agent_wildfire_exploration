import jax
from functools import partial
from jax import grad, jacfwd, vmap, jit, hessian
from jax.lax import scan
import jax.random as jnp_random
import jax.numpy as np

from jax.flatten_util import ravel_pytree

import numpy as onp
from opt_solver import AugmentedLagrangian
from dynamics import SingleIntegrator
from ergodic_metric import ErgodicMetric
from utils import BasisFunc, get_phik, get_ck
from target_distribution import TargetDistribution
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time

class ErgodicTrajectoryOpt(object):
    def __init__(self, initpos, pmap) -> None:
        time_horizon=50
        self.basis           = BasisFunc(n_basis=[5,5])
        self.erg_metric      = ErgodicMetric(self.basis)
        self.robot_model     = SingleIntegrator()
        n,m = self.robot_model.n, self.robot_model.m
        self.target_distr    = TargetDistribution(pmap)
        opt_args = {
            'x0' : np.array(initpos),
            'xf' : np.array(initpos),
            'phik' : get_phik(self.target_distr.evals, self.basis)
        }
        x = np.linspace(opt_args['x0'], opt_args['xf'], time_horizon, endpoint=True)
        u = np.zeros((time_horizon, self.robot_model.m))
        self.init_sol = np.concatenate([x, u], axis=1) 

        @vmap
        def emap(x):
            """ Function that maps states to workspace """
            return np.array([(x[0]+50)/100, (x[1]+50)/100])
        def barrier_cost(e):
            """ Barrier function to avoid robot going out of workspace """
            return (np.maximum(0, e-1) + np.maximum(0, -e))**2
        @jit
        def loss(z, args):
            """ Traj opt loss function, not the same as erg metric """
            x, u = z[:, :n], z[:, n:]
            phik = args['phik']
            e = emap(x)
            ck = get_ck(e, self.basis)
            return 100*self.erg_metric(ck, phik) \
                    + 0.01 * np.mean(u**2) \
                    + np.sum(barrier_cost(e))

        def eq_constr(z, args):
            """ dynamic equality constraints """
            x, u = z[:, :n], z[:, n:]
            x0 = args['x0']
            xf = args['xf']
            return np.vstack([
                x[0]-x0, 
                x[1:,:]-vmap(self.robot_model.f)(x[:-1,:], u[:-1,:]),
                x[-1] - xf
            ])

        def ineq_constr(z,args):
            """ control inequality constraints"""
            x, u = z[:, :n], z[:, n:]
            _g=abs(u)-.05
            #_g=np.zeros((200, 0))
            return _g

        self.solver = AugmentedLagrangian(
                                            self.init_sol,
                                            loss, 
                                            eq_constr, 
                                            ineq_constr, 
                                            opt_args, 
                                            step_size=0.01,
                                            c=1.0
                    )
        # self.solver.solve()