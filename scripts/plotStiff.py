import os
import numpy as np
import pandas
from readConfig import read_config, get_batch_settings

from matplotlib import pyplot as plt
plt.style.use(['science', 'bright'])

from matplotlib import rc
rc('text', usetex=True)

plot_scatter = True

"""
Plotting
"""
exp_folder = '../experiments/3_uni_stiff_conv/'
data_folder = '../data/stiff/'

voids_arr = [1,4,9, 16, 25, 36, 49, 64]
num_samples = 500

experiments = ['58', '65_510']
label = ['GNN 1-9 $n_v$', 'GNN 1-3 $n_v$']

stiffnesses = np.zeros((len(voids_arr), num_samples, 1+len(experiments)))

sqrt_voids = np.sqrt(voids_arr)

sqrt_voids_expanded = np.transpose(np.broadcast_to(sqrt_voids, (num_samples, len(voids_arr))))

for i, voids in enumerate(voids_arr):
    target_file = f'{data_folder}fib{voids}_mesh{num_samples}_t1_uni4/macrostiff.data'
    stiffnesses[i, :, 0] = pandas.read_csv(target_file, sep=' ', header=None).to_numpy().flatten()

    for j, exp in enumerate(experiments):
        if j == 0:
            predic = f'{exp_folder}{exp}/{voids}/stiffnesses_{voids}.txt'
        else:
            predic = f'{exp_folder}{exp}/{voids}/stiffnesses_1.txt'
        stiffnesses[i, :, j+1] = pandas.read_csv(predic, sep=' ', header=None).to_numpy().flatten()

fig, ax = plt.subplots(1,1, figsize=(3.5, 3))
colors = ['gray', '#7570b3', '#d95f02', "#a6761d", "#e7298a", "#66a61e", "#e6ab02"]

move_factor = 0.12
if plot_scatter:
    ax.scatter(sqrt_voids_expanded - 1 * move_factor, stiffnesses[:,:,0], s=4, color=colors[0], alpha=0.1)
ax.plot(sqrt_voids, stiffnesses[:,:,0].mean(axis=1), label='FEM', color=colors[0])

for i in range(len(experiments)):
    if plot_scatter:
        ax.scatter(sqrt_voids_expanded + i * move_factor, stiffnesses[:, :, i+1], s=4, color=colors[i+1], alpha=0.15)
    ax.plot(sqrt_voids, stiffnesses[:, :, i+1].mean(axis=1), label=label[i], color=colors[i+1])

ax.legend(loc='upper right', fontsize=8)

ax.set_xticks(np.unique(sqrt_voids))
ax.set_xticklabels(np.unique(voids_arr), fontsize=12)
ax.set_xlabel(r'Microstructure size $(n_v)$', fontsize=12)
ax.set_ylabel(r'Stiffness $D_{11}$', fontsize=12)
ax.minorticks_off()

# Saving
plt.savefig(f'{exp_folder}/{experiments[-1]}/stiff_reduced.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
plt.savefig(f'{exp_folder}/{experiments[-1]}/stiff_reduced.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')


