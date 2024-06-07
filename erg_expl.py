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
    def __init__(self, initpos, pmap, num_agents, bounds) -> None:
        time_horizon=100
        self.basis           = BasisFunc(n_basis=[8,8])
        self.erg_metric      = ErgodicMetric(self.basis)
        self.robot_model     = SingleIntegrator(num_agents)
        self.target_distr    = TargetDistribution(pmap)
        n,m,N = self.robot_model.n, self.robot_model.m, self.robot_model.N
        opt_args = {
            'x0' : np.array(initpos),
            'xf' : np.array(initpos),
            'phik' : get_phik(self.target_distr.evals, self.basis)
        }
        x = np.linspace(opt_args['x0'], opt_args['xf'], time_horizon, endpoint=True)
        u = np.zeros((time_horizon, N, m))
        self.init_sol = np.concatenate([x, u], axis=2) 

        @vmap
        def emap(x):
            return np.array([(x[0])/(bounds[0,1]-bounds[0,0]), 
                             (x[1])/(bounds[1,1]-bounds[1,0])])
            #return x[:2]

        
        def barrier_cost(e):
            """ Barrier function to avoid robot going out of workspace """
            return (np.maximum(0, e-1) + np.maximum(0, -e))**2
        @jit
        def loss(z, args):
            """ Traj opt loss function, not the same as erg metric """
            x, u = z[:, :, :n], z[:, :, n:]
            phik = args['phik']
            e = emap(x)
            ck = get_ck(e, self.basis)
            return 100*self.erg_metric(ck, phik) \
                    + 0.01 * np.mean(u**2) \
                    + np.sum(barrier_cost(e))

        def eq_constr(z, args):
            """ dynamic equality constraints """
            x, u = z[:, :, :n], z[:, :, n:]
            x0 = args['x0']
            xf = args['xf']
            return np.concatenate([
                (x[0]-x0).flatten(), 
                (x[1:,:]-vmap(self.robot_model.f)(x[:-1,:], u[:-1,:])).flatten(),
                (x[-1] - xf).flatten()
            ])

        def ineq_constr(z):
            x, u = z[:, :, :n], z[:, :, n:]
            e = emap(x)
            ine1 = np.maximum(0, e-1) + np.maximum(0, -e)
            ine2 = np.abs(u) - 20.
            return np.concatenate([ine1.flatten(), ine2.flatten()])

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