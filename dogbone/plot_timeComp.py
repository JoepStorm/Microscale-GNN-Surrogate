"""
Scripts to compare computation time between FEM and GNN methods.

Two plots are made:
- The raw time is the total time, and used in the report
- The alternative we subtract the time spent on computing the material model, which should account for the same time, but does not due to language differences.
"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
plt.style.use(['science', 'bright'])
from matplotlib import rc
rc('text', usetex=True)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

void_array = [1, 4, 9, 16, 25, 36, 49]

# Fem files
fem_time_folder = "fem_combined_times"
fem_folder = "."

# GNN files
gnn_folder = "../experiments/13_comptime/"

# Get FEM times:
fem_times = pd.read_csv(f"{fem_time_folder}/all_times.dat", header=None).to_numpy()
fem_J2_times = pd.read_csv(f"{fem_time_folder}/all_times_J2.dat", header=None).to_numpy()

# Compute net time
net_fem_times = fem_times - fem_J2_times

# Compute mean and std over rows
net_fem_mean = np.mean(net_fem_times, axis=1)
net_fem_std = np.std(net_fem_times, axis=1)

fem_mean = np.mean(fem_times, axis=1)
fem_std = np.std(fem_times, axis=1)

fem_J2_mean = np.mean(fem_J2_times, axis=1)
fem_J2_std = np.std(fem_J2_times, axis=1)

fem_elements = np.zeros(len(void_array))
gnn_J2_times = np.zeros((len(void_array), 5))
gnn_times = np.zeros((len(void_array), 5))
gnn_nostiff_J2_times = np.zeros((len(void_array), 5))
gnn_nostiff_times = np.zeros((len(void_array), 5))

for i, void in enumerate(void_array):
    # Get the number of elements in the micro by looking at the first number in the line after "$Elements"
    track_now = False
    with open(f"{fem_folder}/m{void}_t25/meshes/m_0.msh") as meshdata:
        for k, line in enumerate(meshdata):
            array = line.split()
            if track_now:
                fem_elements[i] = int(array[0])
                break
            if array[0] == "$Elements":
                track_now = True
            else:
                track_now = False

    # get GNN times:
    idx = -1
    with open(f"{gnn_folder}m_{void}/combined_comptime.txt", 'r') as f:
        for k, line in enumerate(f):    # keep iterating, this way it always stores the last line.
            array = line.split()
            if array[0] == 'stress_stiff_tot_time':
                idx += 1
                gnn_times[i, idx] = float(array[1])
            if array[0] == 'stress_stiff_mat_time':
                gnn_J2_times[i, idx] = float(array[1])
            if array[0] == 'stress_tot_time':
                gnn_nostiff_times[i, idx] = float(array[1])
            if array[0] == 'stress_mat_time':
                gnn_nostiff_J2_times[i, idx] = float(array[1])

xlabels = []
for elems, void in zip(fem_elements, void_array):
    xlabels.append(f"{int(elems)}\n{void}")

net_gnn_times = gnn_times - gnn_J2_times
net_gnn_nostiff_times = gnn_nostiff_times - gnn_nostiff_J2_times


# mean and std
net_gnn_times_mean = np.mean(net_gnn_times, axis=1)
net_gnn_times_std = np.std(net_gnn_times, axis=1)

gnn_times_mean = np.mean(gnn_times, axis=1)
gnn_times_std = np.std(gnn_times, axis=1)

net_gnn_nostiff_times_mean = np.mean(net_gnn_nostiff_times, axis=1)
net_gnn_nostiff_times_std = np.std(net_gnn_nostiff_times, axis=1)

gnn_nostiff_times_mean = np.mean(gnn_nostiff_times, axis=1)
gnn_nostiff_times_std = np.std(gnn_nostiff_times, axis=1)

gnn_J2_times_mean = np.mean(gnn_J2_times, axis=1)
gnn_J2_times_std = np.std(gnn_J2_times, axis=1)

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(3.5, 2.8), height_ratios=[3, 1], sharex=True)

ax[0].plot(fem_elements, net_fem_mean, label='FEM')
ax[0].fill_between(fem_elements, net_fem_mean - 1.96 * net_fem_std, net_fem_mean + 1.96 * net_fem_std, alpha=0.3, color=colors[0], edgecolor='none')

ax[0].plot(fem_elements, net_gnn_times_mean, label=r'GNN $\boldsymbol{\sigma}^{\Omega}, \mathbf{D}$')
ax[0].fill_between(fem_elements, net_gnn_times_mean - 1.96 * net_gnn_times_std, net_gnn_times_mean + 1.96 * net_gnn_times_std, alpha=0.3, color=colors[1], edgecolor='none')
ax[0].plot(fem_elements, net_gnn_nostiff_times_mean, label=r'GNN $\boldsymbol{\sigma}^{\Omega}$')
ax[0].fill_between(fem_elements, net_gnn_nostiff_times_mean - 1.96 * net_gnn_nostiff_times_std, net_gnn_nostiff_times_mean + 1.96 * net_gnn_nostiff_times_std, alpha=0.3, color=colors[2], edgecolor='none')

# Plots with total time dashed
ax[0].plot(fem_elements, fem_mean, linestyle='dashed', color=colors[0], alpha=0.5, label='J2')
ax[0].plot(fem_elements, gnn_times_mean, linestyle='dashed', color=colors[1], alpha=0.5)
ax[0].plot(fem_elements, gnn_nostiff_times_mean, linestyle='dashed', color=colors[2], alpha=0.5)
ax[1].plot(fem_elements, gnn_nostiff_times_mean, linestyle='dashed', color=colors[2], alpha=0.5)



ax[0].set_ylabel("Comp. time - J2 [sec]")
ax[0].yaxis.set_label_coords(-0.13, 0.2)
ax[0].legend()
ax[0].set_ylim(ymin=0)
fig.subplots_adjust(hspace=0.05)

ax[1].plot(fem_elements, net_gnn_nostiff_times_mean, label=r'GNN $\boldsymbol{\sigma}^{\Omega}$', color=colors[2])
ax[1].fill_between(fem_elements, net_gnn_nostiff_times_mean - 1.96 * net_gnn_nostiff_times_std, net_gnn_nostiff_times_mean + 1.96 * net_gnn_nostiff_times_std, alpha=0.3, color=colors[2], edgecolor='none')

ax[1].set_xlim(xmin=0)
ax[1].set_ylim([29, 38])
ax[1].set_xticks(fem_elements)
ax[1].set_xticklabels(xlabels)
ax[1].set_xlabel("Num. Elements\n Num. Voids")
ax[1].xaxis.set_label_coords(-0.3, -0.1)
ax[1].set_yticks([30, 33, 36])
ax[1].minorticks_off()

# ax[1].legend()

plt.savefig(f'comp_time.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
plt.savefig(f'comp_time.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')



# Same but now raw time
fig, ax = plt.subplots(1, 1, figsize=(3, 2))

ax.plot(fem_elements, fem_mean, label='FEM')
ax.fill_between(fem_elements, fem_mean - 1.96 * fem_std, fem_mean + 1.96 * fem_std, alpha=0.3, color=colors[0], edgecolor='none')

ax.plot(fem_elements, gnn_times_mean, label=r'GNN $\boldsymbol{\sigma}^{\Omega}, \mathbf{D}$')
ax.fill_between(fem_elements, gnn_times_mean - 1.96 * gnn_times_std, gnn_times_mean + 1.96 * gnn_times_std, alpha=0.3, color=colors[1], edgecolor='none')
ax.plot(fem_elements, gnn_nostiff_times_mean, label=r'GNN $\boldsymbol{\sigma}^{\Omega}$')
ax.fill_between(fem_elements, gnn_nostiff_times_mean - 1.96 * gnn_nostiff_times_std, gnn_nostiff_times_mean + 1.96 * gnn_nostiff_times_std, alpha=0.3, color=colors[2], edgecolor='none')

ax.set_ylabel("Computation time [sec]")
ax.yaxis.set_label_coords(-0.17, 0.48)
ax.legend()
ax.set_ylim(ymin=0)
ax.set_xlim(xmin=0)
ax.set_xticks(fem_elements)
ax.set_xticklabels(xlabels)
ax.set_xlabel("Num. Elements\n Num. Voids")
ax.xaxis.set_label_coords(-0.3, -0.03)
ax.tick_params(axis='x', which='minor', bottom=False, top=False)

plt.savefig(f'comp_time_raw.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
plt.savefig(f'comp_time_raw.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')

