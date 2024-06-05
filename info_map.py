import jax.numpy as np
from numpy import random
import matplotlib.pyplot as plt

empty_grid = np.ones((10000, 10000))


def gaussian(x, sigma):
    c = np.sqrt(2 * np.pi)
    return np.exp(-0.5 * (x / sigma)**2) / sigma / c


x_val = np.linspace(-3, 3, 100)
y_val = gaussian(x_val, 1).reshape((10, 10))
x_val.reshape(10, 10)

plt.plot(x_val, y_val)
plt.show()

print(gaussian(0, 1, 1))