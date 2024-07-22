import jax.numpy as np
import numpy as onp
from gaussian import gaussian
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def dynamic_info_init(num_targets, size):
    pos = np.floor(onp.random.uniform(0, size, 2*num_targets))
    pmap = gaussian(size, pos[0], pos[1], 5)
    peak_indices = [np.where(pmap>.1)]
    for i in range(1, num_targets):
        new_peak = gaussian(size, pos[2*i], pos[2*i+1], 5)
        pmap += new_peak
        peak_indices.append(np.where(new_peak>.1))
    pos = np.reshape(pos, (num_targets, 2))

    vel = onp.random.uniform(onp.random.uniform(0, 20, 2*num_targets))
    vel = np.reshape(vel, (num_targets, 2))
    return pmap, peak_indices, pos, vel

def dynamic_info_step(num_targets, size, map=None, target_pos=None, vel=None):
    if map==None:
        pmap, peak_indices, target_pos, vel = dynamic_info_init(num_targets, size)
    else:
        target_pos = np.clip(target_pos + vel, 0, size)
        zero_indices = np.where(target_pos==0)
        max_indices = np.where(target_pos==size)
        vel[zero_indices] *= -1
        vel[max_indices] *= -1

        pmap = gaussian(size, target_pos[0][0], target_pos[0][1], 5)
        peak_indices = [np.where(pmap>.1)]
        for i in range(1, num_targets):
            new_peak = gaussian(size, target_pos[i][0], target_pos[i][1], 5)
            pmap += new_peak
            peak_indices.append(np.where(new_peak>.1))
    
    return pmap, peak_indices, target_pos, vel

'''
map, peak_idx, target_pos, target_vel = dynamic_info_init(5, 100)
map_1, peak_idx_1, target_pos_1, target_vel_1 = dynamic_info_step(5, 100, map, target_pos, target_vel)
map_2, peak_idx_2, target_pos_2, target_vel_2 = dynamic_info_step(5, 100, map_1, target_pos_1, target_vel_1)
map_3, peak_idx_3, target_pos_3, target_vel_3 = dynamic_info_step(5, 100, map_2, target_pos_2, target_vel_2)
map_4, peak_idx_4, target_pos_4, target_vel_4 = dynamic_info_step(5, 100, map_3, target_pos_3, target_vel_3)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.imshow(map_1, origin='lower')
ax2.imshow(map_2, origin='lower')
ax3.imshow(map_3, origin='lower')
ax4.imshow(map_4, origin='lower')

plt.show()
'''