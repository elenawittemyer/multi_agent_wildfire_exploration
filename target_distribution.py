import jax.numpy as np
import numpy as onp
from jax import vmap, jit
from gaussian import gaussian_1d, gaussian
import matplotlib.pyplot as plt

class TargetDistribution(object):
    def __init__(self, pmap, size) -> None:
        self.n = 2
        self.domain = np.meshgrid(
            *[np.linspace(0.01, 0.99, size)]*self.n 
        )
        pmap=pmap.reshape((1, size**2))
        pmap=pmap[0]
        self._s = np.stack([X.ravel() for X in self.domain]).T
        self.evals = (
            self.p(pmap) , self._s
        )

    def p(self, map):
        return map

    def update(self):
        pass

def pdf(V, x, args):
    frame = args['frame']
    size = args['size']

    den = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy')
    vis = np.maximum(1-den, np.zeros(den.shape))
    C = vis[x[0], x[1]]
    
    v_true = V[int(len(V)/2)]
    spread =100/v_true
    rad = spread/2 # smoke vs peak effect seems balanced?
    pdf_V = []
    gauss_peak = gaussian_1d(200, C*100, rad)

    for v in V:
        pdf_V.append(gauss_peak[(np.int32(v*spread))])
    return np.array(pdf_V)

def _shannon_entropy(V, x, map_args):
    pdf_array = pdf(V, x, map_args)
    return -1*np.sum(pdf_array * np.log(pdf_array)/np.log(2))

shannon_entropy = vmap(_shannon_entropy, in_axes=(0, 0, None))

def calc_entropy(map, size, frame):
    args = {
    'frame': frame,
    'size': size
    }

    X,Y = onp.meshgrid(onp.arange(args['size']), onp.arange(args['size']))
    out = onp.column_stack((Y.ravel(), X.ravel()))

    def _eval_map(x, map):
        return map[x[0], x[1]]
    eval_map = vmap(_eval_map, in_axes=(0, None))

    v_array = eval_map(out, map)

    def _measure_noise(v):
        noise = v/10
        return np.linspace(v-noise, v+noise, 10)
    measure_noise = vmap(_measure_noise, in_axes=0)

    V_array = measure_noise(v_array)

    info_grid = shannon_entropy(V_array, out, args)
    info_grid = info_grid.reshape((args['size'], args['size']))
    return info_grid