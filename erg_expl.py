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
    def __init__(self, initpos, pmap, num_agents, size, f) -> None:
        time_horizon=40
        self.basis           = BasisFunc(n_basis=[5,5])
        self.erg_metric      = ErgodicMetric(self.basis)
        self.robot_model     = SingleIntegrator(num_agents)
        n,m,N = self.robot_model.n, self.robot_model.m, self.robot_model.N
        self.target_distr    = TargetDistribution(pmap, size)
        opt_args = {
            'x0' : initpos,
            'xf' : initpos,
            #'xf' : np.zeros((N, 2)),
            'phik' : get_phik(self.target_distr.evals, self.basis)
        }
        ''' Initialize state '''
        x = np.linspace(opt_args['x0'], opt_args['xf'], time_horizon, endpoint=True)
        u = np.zeros((time_horizon, N, m))
        self.init_sol = np.concatenate([x, u], axis=2) 
        def _emap(x):
            ''' Map state space to exploration space '''
            return np.array([(x+(size/2))/size])
        emap = vmap(_emap, in_axes=0)

        def barrier_cost(e):
            """ Barrier function to avoid robot going out of workspace """
            return (np.maximum(0, e-1) + np.maximum(0, -e))**2
        @jit
        def loss(z, args):
            """ Traj opt loss function, not the same as erg metric """
            x, u = z[:, :, :n], z[:, :, n:]
            phik = args['phik']
            e = np.squeeze(emap(x))
            ck = np.mean(vmap(get_ck, in_axes=(1, None))(e, self.basis), axis=0)
            erg_m = self.erg_metric(ck, phik)
            return 100 * N * erg_m \
                    + .1 * np.mean(u**2) \
                    + 10 * np.sum(barrier_cost(e))
        def eq_constr(z, args):
            """ dynamic equality constriants """
            x, u = z[:, :, :n], z[:, :, n:]
            x0 = args['x0']
            xf = args['xf']
            return np.concatenate([
                (x[0]-x0).flatten(), 
                (x[1:,:]-vmap(self.robot_model.f)(x[:-1,:], u[:-1,:])).flatten(),
                (x[-1] - xf).flatten()
            ])

        def ineq_constr(z,args):
            """ control inequality constraints"""
            x, u = z[:, :, :n], z[:, :, n:]
            #_g = np.max(np.array([abs(u)-10, .5-abs(u)]))
            _g =  abs(u)-10
            return _g

        self.solver = AugmentedLagrangian(
                                            self.init_sol,
                                            loss, 
                                            eq_constr, 
                                            ineq_constr,
                                            f, 
                                            opt_args, 
                                            step_size=0.01,
                                            c=1.0
                    )
        # self.solver.solve()
