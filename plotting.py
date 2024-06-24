import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def get_colormap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def final_plot(x, map_i, map_f, N, t_f):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(map_f, origin="lower")
    cmap = get_colormap(N+1)
    map_size = map_i.shape[0]

    starts = []
    for i in range(N):
        ax1.plot(np.array(x[i][0]).flatten(), np.array(x[i][1]).flatten(), c=cmap(i))
        starts.append(plt.Circle(((np.array(x[i][0]).flatten()[0], np.array(x[i][1]).flatten()[0])), .3, color='w'))
        ax1.add_patch(starts[i])
    
    ax2.imshow(map_i, origin="lower")
    smoke_grid = np.load('smoke_density/smoke_grid_' + str(map_size) + '/smoke_array_' + str(t_f) + '.npy')
    ax1.imshow(smoke_grid, vmin=0, vmax=1, alpha = .5, cmap=plt.cm.gray, interpolation='nearest', origin='lower')

    plt.show()

def animate_plot(map_size, t_f, path):
    path_x = np.array(path[0][0]).flatten()  ##TODO: CURRENTLY ONLY FOR 1 AGENT
    path_y = np.array(path[0][1]).flatten()

    fig, ax = plt.subplots()
    den = np.load('smoke_density/smoke_grid_' + str(map_size) + '/smoke_array_0.npy')
    line = [[path_x[0], path_x[1]], [path_y[0], path_y[1]]]
    img = ax.imshow(den, origin='lower', animated = True)
    traj, = ax.plot(line[0], line[1], c='w')
    
    def updatefig(frame, img, traj, ax):
        den = np.load('smoke_density/smoke_grid_' + str(map_size) + '/smoke_array_' + str(frame) + '.npy')
        line = [[path_x[frame], path_x[frame+1]], [path_y[frame], path_y[frame+1]]]
        img.set_data(den)
        traj, = ax.plot(line[0],line[1], c='w')
        return img, traj

    ani = animation.FuncAnimation(fig, updatefig, frames=t_f-1, fargs=(img, traj, ax), interval=1, blit=True)
    mywriter = animation.FFMpegWriter(fps = 10)
    ani.save("videos/smoke_one_agent.mp4", writer=mywriter)
    plt.close(fig)

def smoke_vs_info(map_size, t_f, path, init_map, num_agents):
    time_map = np.zeros((map_size, map_size))
    smoke_sum = np.zeros((map_size, map_size))
    for frame in range(t_f):
        smoke_sum += np.load('smoke_density/smoke_grid_' + str(map_size) + '/smoke_array_' + str(frame) + '.npy')
    smoke_avg = smoke_sum / t_f
    for agent in range(num_agents):
        x_cells = np.floor(np.array(path[agent][0]).flatten()).astype(int)
        y_cells = np.floor(np.array(path[agent][1]).flatten()).astype(int)
        time_map[x_cells, y_cells] += 1

    norm_time_map = np.divide(time_map, init_map)
    vis_bins = np.arange(0, 1, .1)
    freq = []
    for i in range(len(vis_bins)-1):
        indices = np.where(np.all(np.array([vis_bins[i]<=smoke_avg, vis_bins[i+1]>smoke_avg]), axis=0))
        freq.append(np.sum(norm_time_map[indices]))
    freq = np.array(freq)
    plt.stairs(freq, vis_bins)
    plt.show()

def time_dstrb_comp(map_size, t_f, init_map, path_ns, path_s, num_agents, f_map_s, f_map_ns): #TODO: only plotting one trajectory?
    peaks = np.where(init_map>.02)
    time_map_s = np.zeros((map_size, map_size))
    time_map_ns = np.zeros((map_size, map_size))

    smoke_sum = np.zeros((map_size, map_size))
    for frame in range(t_f):
        smoke_sum += np.load('smoke_density/smoke_grid_' + str(map_size) + '/smoke_array_' + str(frame) + '.npy')
    avg_den = smoke_sum/np.max(smoke_sum)

    for agent in range(num_agents):
        x_cells_s = np.floor(np.array(path_s[agent][0]).flatten()).astype(int)
        y_cells_s = np.floor(np.array(path_s[agent][1]).flatten()).astype(int)
        time_map_s[x_cells_s, y_cells_s] += 1
        
        x_cells_ns = np.floor(np.array(path_ns[agent][0]).flatten()).astype(int)
        y_cells_ns = np.floor(np.array(path_ns[agent][1]).flatten()).astype(int)
        time_map_ns[x_cells_ns, y_cells_ns] += 1

    freq_s = np.sum(time_map_s[peaks])/np.sum(time_map_s)
    freq_ns = np.sum(time_map_ns[peaks])/np.sum(time_map_ns)

    print('% time spent at peaks- smoke: ' + str(freq_s))
    print('% time spent at peaks- no smoke: ' + str(freq_ns))

    fig, axs = plt.subplots(2, 2)
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

    cmap = get_colormap(num_agents+1)
    
    ax1.imshow(init_map, origin='lower')
    ax2.imshow(init_map, origin='lower')

    for i in range(num_agents):
        ax1.plot(np.array(path_s[i][0]).flatten(), np.array(path_s[i][1]).flatten(), c=cmap(i))
        ax2.plot(np.array(path_ns[i][0]).flatten(), np.array(path_ns[i][1]).flatten(), c=cmap(i))

    ax1.imshow(avg_den, vmin=0, vmax=1, alpha = .5, cmap=plt.cm.gray, interpolation='nearest', origin='lower')
    ax3.imshow(f_map_s, origin='lower')
    ax4.imshow(f_map_ns, origin='lower')
    
    plt.show()
    
