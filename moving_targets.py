import jax.numpy as np
import numpy as onp
from gaussian import gaussian, gaussian_measurement
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def dynamic_info_init(size, num_targets):
    pos = np.floor(onp.random.uniform(5, size-5, 2*num_targets))
    pmap = gaussian(size, pos[0], pos[1], 5)
    peak_indices = [np.where(pmap>.1)]
    for i in range(1, num_targets):
        new_peak = gaussian(size, pos[2*i], pos[2*i+1], 5)
        pmap += new_peak
        peak_indices.append(np.where(new_peak>.1))
    pos = np.reshape(pos, (num_targets, 2))

    vel = onp.random.uniform(onp.random.uniform(0, 10, 2*num_targets))
    vel = np.reshape(vel, (num_targets, 2))
    return pmap, peak_indices, pos, vel

def dynamic_info_step(num_targets, size, pmap=None, peak_indices = None, target_pos=None, vel=None):
    if pmap==None:
        pmap, peak_indices, target_pos, vel = dynamic_info_init(num_targets, size)
    else:
        '''
        new_map = np.zeros((size, size))
        new_peak_idxs = []
        target_pos = np.clip(target_pos + vel, 2, size-2)
    
        for k in range(num_targets):
            new_peak_idx_x = np.round(np.clip(peak_indices[k][0]+vel[k][0], 2, size-2)).astype(int)
            new_peak_idx_y = np.round(np.clip(peak_indices[k][1]+vel[k][1], 2, size-2)).astype(int)
            new_peak_idx = (new_peak_idx_x, new_peak_idx_y)
            new_map = new_map.at[new_peak_idx].set(pmap[peak_indices[k]])
            new_peak_idxs.append(new_peak_idx)
        
        zero_indices = np.where(target_pos==2)
        max_indices = np.where(target_pos==size-2)
        vel[zero_indices] *= -1
        vel[max_indices] *= -1

        return new_map, new_peak_idxs, target_pos, vel

        '''
        static_targets = onp.random.choice(2, num_targets, p=[0.5, 0.5])
        temp_vel = np.multiply(vel, np.hstack((np.array([static_targets]).T, np.array([static_targets]).T)))

        new_map = np.zeros((size, size))
        new_peak_idx = []

        target_pos = np.clip(target_pos + temp_vel, 2, size-2)
        zero_indices = np.where(target_pos==2)
        max_indices = np.where(target_pos==size-2)
        vel[zero_indices] *= -1
        vel[max_indices] *= -1

        for k in range(len(static_targets)):
            if static_targets[k]==0:
                new_map = new_map.at[peak_indices[k]].set(pmap[peak_indices[k]])
                new_peak_idx.append(peak_indices[k])
            else:
                new_peak = gaussian(size, target_pos[k][0], target_pos[k][1], 5)
                new_map += new_peak
                new_peak_idx.append(np.where(new_peak>.1))

    return new_map, new_peak_idx, target_pos, vel
    

'''
map, peak_idx, target_pos, target_vel = dynamic_info_init(5, 100)
map += gaussian_measurement(100, np.floor(target_pos[0][0]).astype(int), np.floor(target_pos[0][1]).astype(int), .03)
map_1, peak_idx_1, target_pos_1, target_vel_1 = dynamic_info_step(5, 100, map, peak_idx, target_pos, target_vel)
map_2, peak_idx_2, target_pos_2, target_vel_2 = dynamic_info_step(5, 100, map_1, peak_idx_1, target_pos_1, target_vel_1)
map_3, peak_idx_3, target_pos_3, target_vel_3 = dynamic_info_step(5, 100, map_2, peak_idx_2, target_pos_2, target_vel_2)
map_4, peak_idx_4, target_pos_4, target_vel_4 = dynamic_info_step(5, 100, map_3, peak_idx_3, target_pos_3, target_vel_3)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.imshow(map_1, origin='lower')
ax2.imshow(map_2, origin='lower')
ax3.imshow(map_3, origin='lower')
ax4.imshow(map_4, origin='lower')

plt.show()
'''
