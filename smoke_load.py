import numpy as np
import matplotlib.pyplot as plt

smoke = np.load('smoke_density/smoke_array.npy')
im = plt.imshow(smoke, vmin=0, vmax=1, cmap=plt.cm.gray,
                    interpolation='nearest', origin='lower')

plt.show()