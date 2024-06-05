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
from fn_info_map import initmap

def initpos(num):
    poslist=[[0,0], [4,-4], [8,-8], [12,-12]]
    return poslist[num]
    
class ErgodicTrajectoryOpt(object):
    def __init__(self, initpos, pmap, type) -> None:
        if type=="t":
            time_horizon=48
            self.basis           = BasisFunc(n_basis=[5,5])
            self.erg_metric      = ErgodicMetric(self.basis)
            self.robot_model     = SingleIntegrator()
            n,m = self.robot_model.n, self.robot_model.m
            self.target_distr    = TargetDistribution(pmap, type)
            opt_args = {
                'x0' : np.array(initpos),
                'xf' : np.array(initpos),
                'phik' : get_phik(self.target_distr.evals, self.basis)
            }
            x = np.ones((time_horizon, self.robot_model.n)) * opt_args['x0']
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
                """ dynamic equality constriants """
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
                _g=abs(u)-7
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
        if type=="c":
            time_horizon=6
            self.basis           = BasisFunc(n_basis=[5,5])
            self.erg_metric      = ErgodicMetric(self.basis)
            self.robot_model     = SingleIntegrator()
            n,m = self.robot_model.n, self.robot_model.m
            self.target_distr    = TargetDistribution(pmap, type)
            opt_args = {
                'x0' : np.array(initpos),
                'xf' : np.array([110, -20]),
                'phik' : get_phik(self.target_distr.evals, self.basis)
            }
            x = np.ones((time_horizon, self.robot_model.n)) * opt_args['x0']
            x = np.linspace(opt_args['x0'], opt_args['xf'], time_horizon, endpoint=True)
            u = np.zeros((time_horizon, self.robot_model.m))
            self.init_sol = np.concatenate([x, u], axis=1) 

            @vmap
            def emap(x):
                """ Function that maps states to workspace """
                return np.array([(x[0]+135)/270, (x[1]+90)/270])
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
                """ dynamic equality constriants """
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
                _g=abs(u)-70
                #_g=np.zeros((time_horizon, 0))
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


xpoints=onp.linspace(-49,50,100)
ypoints=onp.linspace(-49,50,100)

'''
traj_opt = ErgodicTrajectoryOpt([0,0])
X, Y = traj_opt.target_distr.domain
P = traj_opt.target_distr.evals[0].reshape(X.shape)

for _ in range(1000):
    traj_opt.solver.solve(max_iter=50)
    sol = traj_opt.solver.get_solution()
    clear_output(wait=True)
    #plt.contour(X, Y, P)
    if _%30==0:
        plt.contour(xpoints, ypoints, initmap)
        plt.plot(sol['x'][:,0], sol['x'][:,1])

        # plt.xlim(0,1)
        # plt.ylim(0,1)
        plt.axis('equal')
        plt.show()
'''