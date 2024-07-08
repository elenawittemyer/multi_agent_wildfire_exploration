import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

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
    path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    smoke_grid = np.load(path + '/smoke_density/smoke_grid_' + str(map_size) + '/smoke_array_' + str(t_f) + '.npy')
    ax1.imshow(smoke_grid, vmin=0, vmax=1, alpha = .5, cmap=plt.cm.gray, interpolation='nearest', origin='lower')

    plt.show()

def animate_plot(map_size, t_f, path, num_agents, map_i):
    cmap = get_colormap(num_agents+1)
    
    path_x = []
    path_y = []
    for i in range(num_agents):
        path_x.append(np.array(path[i][0]).flatten()) 
        path_y.append(np.array(path[i][1]).flatten())

    fig, ax = plt.subplots()
    path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    den = np.load(path + '/smoke_density/smoke_grid_' + str(map_size) + '/smoke_array_0.npy')
    ax.imshow(map_i, origin="lower")
    img = ax.imshow(den, vmin=0, vmax=1, alpha = .5, cmap=plt.cm.gray, interpolation='nearest', origin='lower', animated = True)
    for i in range(num_agents):
        line = [[path_x[i][0], path_x[i][1]], [path_y[i][0], path_y[i][1]]]
        traj, = ax.plot(line[0], line[1], c=cmap(i))
    
    def updatefig(frame, img, traj, ax):
        den = np.load(path + '/smoke_density/smoke_grid_' + str(map_size) + '/smoke_array_' + str(frame) + '.npy')
        img.set_data(den)
        for i in range(num_agents):
            line = [[path_x[i][frame], path_x[i][frame+1]], [path_y[i][frame], path_y[i][frame+1]]]
            traj, = ax.plot(line[0],line[1], c=cmap(i))
        return img, traj

    ani = animation.FuncAnimation(fig, updatefig, frames=t_f-1, fargs=(img, traj, ax), interval=1, blit=True)
    mywriter = animation.FFMpegWriter(fps = 10)
    ani.save(path + "/videos/smoke_multi_agent.mp4", writer=mywriter)
    plt.close(fig)

def animate_vis(map_size, t_f, map_i, peak_idx):

    def blackout_map(map, peak_indices, size, frame):
        local_map = np.copy(map)
        den_cutoff = .45
        den = np.load('data_and_plotting/smoke_density/smoke_grid_' + str(size) + '/smoke_array_' + str(frame) + '.npy')
        den_avg = []
        for i in range(len(peak_indices)):
            den_avg.append(np.average(den[peak_indices[i]]))
        den_avg = np.array(den_avg)

        blackout_array = den_avg<den_cutoff 
        blackout_array = blackout_array * 1
        for i in range(len(peak_indices)):
            local_map[peak_indices[i]] *= blackout_array[i]
        return local_map
    
    fig, ax = plt.subplots()
    path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    den = np.load(path + '/smoke_density/smoke_grid_' + str(map_size) + '/smoke_array_0.npy')
    b_map = blackout_map(map_i, peak_idx, map_size, 0)
    img1 = ax.imshow(b_map, origin="lower")
    img2 = ax.imshow(den, vmin=0, vmax=1, alpha = .5, cmap=plt.cm.gray, interpolation='nearest', origin='lower', animated = True)

    def updatefig(frame, img1, img2):
        den = np.load(path + '/smoke_density/smoke_grid_' + str(map_size) + '/smoke_array_' + str(frame) + '.npy')
        b_map = blackout_map(map_i, peak_idx, map_size, frame)
        img1.set_data(b_map)
        img2.set_data(den)
        return img1, img2
    
    ani = animation.FuncAnimation(fig, updatefig, frames=t_f-1, fargs=(img1, img2), interval=1, blit=True)
    mywriter = animation.FFMpegWriter(fps = 10)
    ani.save(path + "/videos/smoke_info_visibility.mp4", writer=mywriter)
    plt.close(fig)


def smoke_vs_info(map_size, t_f, path, init_map, num_agents):
    time_map = np.zeros((map_size, map_size))
    smoke_sum = np.zeros((map_size, map_size))
    path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    for frame in range(t_f):
        smoke_sum += np.load(path + '/smoke_density/smoke_grid_' + str(map_size) + '/smoke_array_' + str(frame) + '.npy')
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
    path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    for frame in range(t_f):
        smoke_sum += np.load(path + '/smoke_density/smoke_grid_' + str(map_size) + '/smoke_array_' + str(frame) + '.npy')
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
    
def plot_ergodic_metric():
    path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(path + '/plotting_data/erg_metric_data.txt', 'r') as file:
        erg_vals = np.array(file.read().splitlines()).astype(float)
    time = range(len(erg_vals))

    erg_diff = np.diff(erg_vals)
    peaks = np.where(erg_diff>.1)[0].tolist()
    peaks.insert(0, 0)
    peaks.append(len(erg_diff))

    plateaus = []
    avg_erg = []
    for i in range(len(peaks)-1):
        plateau_region = np.where(np.abs(erg_diff)<.03)[0]
        plateau_region = plateau_region[np.logical_and(peaks[i]<plateau_region, plateau_region<peaks[i+1])]
        avg_erg.append(np.average(erg_vals[plateau_region]))
        plateaus.append(plateau_region[0])
    plateaus = np.array(plateaus)
    avg_erg = np.array(avg_erg)

    min_time = plateaus-peaks[0:len(plateaus)]
    avg_min_time = np.sum(min_time)/len(min_time)
    print("Avg erg metric minimization time: " + str(avg_min_time))
    print("Avg final erg metric: " + str(np.average(avg_erg)))

    plt.plot(time, erg_vals)
    plt.vlines(plateaus, 0, np.max(erg_vals), colors='r', linestyles='dashed')
    plt.xlabel('Iterations')
    plt.ylabel('Ergodic Metric')
    plt.show()

def plot_info_reduct(t_f):
    path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(path + '/plotting_data/info_map_data.txt', 'r') as file:
        info_sum = np.array(file.read().splitlines()).astype(float)
    time = np.arange(0, t_f, t_f/len(info_sum))
    print("Info reduction: " + str((1-info_sum[-1]/info_sum[0])*100) + str('%'))
    plt.plot(time, info_sum)
    plt.xlabel('Time')
    plt.ylabel('Total Map Uncertainty')
    plt.show()
