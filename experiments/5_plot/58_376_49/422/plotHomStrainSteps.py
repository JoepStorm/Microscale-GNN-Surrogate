import os
import numpy as np
import pandas

from matplotlib import pyplot as plt
plt.style.use(['science', 'bright'])

from PyPDF2 import PdfMerger
from matplotlib import rc
rc('text', usetex=True)

"""
Plotting
"""
plot_type = 'separate'  # 'combined' or 'separate'

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# colors = ['#1b9e77', '#d95f02', '#7570b3', "#a6761d", "#e7298a", "#66a61e", "#e6ab02"]
linestyles = ['-', ':', '--', '-.', ':']

data = []
data.append(pandas.read_csv('sig_eps_422', header=None, sep=' ').to_numpy())


timesteps = 26
merger_hom = PdfMerger()  # initialize merger

# Plot one additional point, for both the data [:,0] and [:,1] in each step
for t in range(timesteps):
    fig, ax = plt.subplots(1, len(data), figsize=(.5 + 2.5 * len(data), 2))

    # First component
    # Print true
    ax.plot( np.append([0], data[0][:t,0]), np.append([0], data[0][:t,1]), colors[0], alpha=0.7, linestyle=linestyles[0], label='True')
    # Print pred
    ax.plot( np.append([0], data[0][:t,0]), np.append([0], data[0][:t,2]), colors[0], linestyle=linestyles[1], label='Pred')

    # Second component
    ax.plot( np.append([0], data[0][:t,3]), np.append([0], data[0][:t,4]), colors[1], alpha=0.7, linestyle=linestyles[0])
    ax.plot( np.append([0], data[0][:t,3]), np.append([0], data[0][:t,5]), colors[1], linestyle=linestyles[1])

    # Third component
    ax.plot( np.append([0], data[0][:t,6]), np.append([0], data[0][:t,7]), colors[2], alpha=0.7, linestyle=linestyles[0])
    ax.plot( np.append([0], data[0][:t,6]), np.append([0], data[0][:t,8]), colors[2], linestyle=linestyles[1])

    ax.set_xlabel(r'$\varepsilon^{\Omega}$ $[-]$')
    ax.set_ylabel(r'$\sigma^{\Omega}$ $[MPa]$')
    ax.set_ylim([-10.9, 11.2])
    ax.set_xlim([-.0249, 0.0222])

    # Add floating text
    ax.text(0.12, 0.19, "$x$", transform=ax.transAxes, fontsize=12, color=colors[0])
    ax.text(0.12, 0.12, "$y$", transform=ax.transAxes, fontsize=12, color=colors[1])
    ax.text(0.12, 0.35, "$xy$", transform=ax.transAxes, fontsize=12, color=colors[2])

    # remove minor ticks:
    # ax[i].xaxis.set_tick_params(which='minor', bottom=False, top=False)
    # ax[i].yaxis.set_tick_params(which='minor', left=False, right=False)

    ax.legend(loc='upper left', fontsize=8)

    # plt.tight_layout()
    filename = f'422_hom_{t}.pdf'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
    plt.close()
    merger_hom.append(filename)
    os.remove(filename)

merger_hom.write(f"422_hom_timesteps.pdf")
merger_hom.close()
