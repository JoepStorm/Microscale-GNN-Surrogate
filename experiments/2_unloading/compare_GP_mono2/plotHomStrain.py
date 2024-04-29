import os
import numpy as np
import pandas

from matplotlib import pyplot as plt
plt.style.use(['science', 'bright'])

from matplotlib import rc
rc('text', usetex=True)

"""
Plotting
"""
plot_type = 'combined'  # 'combined' or 'separate'

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# colors = ['#1b9e77', '#d95f02', '#7570b3', "#a6761d", "#e7298a", "#66a61e", "#e6ab02"]
linestyles = ['-', ':', '--', '-.', ':', '--']


gp_data = pandas.read_csv('58_sig_eps_1155', header=None, sep=' ').to_numpy()
mono_data = pandas.read_csv('69_sig_eps_1155', header=None, sep=' ').to_numpy()
nomat_data = pandas.read_csv('64_sig_eps_1155', header=None, sep=' ').to_numpy()
noepspeq_data = pandas.read_csv('68_570_sig_eps_1155', header=None, sep=' ').to_numpy()
fullstrain_data = pandas.read_csv('68_572_sig_eps_1155', header=None, sep=' ').to_numpy()

eps_mono = mono_data[:,::3]
eps_gp = gp_data[:,::3]
assert np.all(np.isclose(eps_mono,eps_gp)), "eps not equal"
sig_mono = mono_data[:,1::3]
sig_gp = gp_data[:,1::3]
assert np.all(np.isclose(sig_mono, sig_gp)), "sig not equal"

pred_mono = mono_data[:,2::3]
pred_gp = gp_data[:,2::3]
pred_nomat = nomat_data[:,2::3]
pred_noepspeq = noepspeq_data[:,2::3]
pred_fullstrain = fullstrain_data[:,2::3]

components = 3

if plot_type == 'combined':
    # Plotting
    fig, ax = plt.subplots(1,1,figsize=(3,2.5))
    # true
    ax.plot(np.append([0], eps_mono[:, 0]), np.append([0],sig_mono[:, 0]), 'k', alpha=0.4, linestyle=linestyles[0], label='True')
    ax.plot(np.append([0], eps_mono[:, 0]), np.append([0], pred_mono[:, 0]), colors[0], linestyle=linestyles[1], label='Mono')
    ax.plot(np.append([0], eps_gp[:, 0]), np.append([0], pred_gp[:, 0]), colors[1], linestyle=linestyles[2], label='GP')
    # ax.plot(np.append([0], eps_mono[:, 0]), np.append([0], pred_nomat[:, 0]), colors[2], linestyle=linestyles[3], label='No Mat')

    for i in range(components-1):
        ax.plot(np.append([0], eps_mono[:,i+1]), np.append([0], sig_mono[:,i+1]), 'k', alpha=0.4, linestyle=linestyles[0])
        ax.plot(np.append([0], eps_mono[:,i+1]), np.append([0], pred_mono[:,i+1]), colors[0], linestyle=linestyles[1])
        ax.plot(np.append([0], eps_gp[:,i+1]), np.append([0], pred_gp[:,i+1]), colors[1], linestyle=linestyles[2])
        # ax.plot(np.append([0], eps_mono[:, i+1]), np.append([0], pred_nomat[:, i+1]), colors[2], linestyle=linestyles[3])

    ax.set_xlabel(r'$\varepsilon^{\Omega}$')
    ax.set_ylabel(r'$\sigma^{\Omega}$')
    ax.legend(loc='lower right')
    # ax.set_ylim([-1, 13.5])
    # ax.set_xlim([-.005, 0.06])

    # remove minor ticks:
    # ax.xaxis.set_tick_params(which='minor', bottom=False, top=False)
    # ax.yaxis.set_tick_params(which='minor', left=False, right=False)

    plt.tight_layout()
    # plt.savefig(f'compare_GP_mono_true.pdf')
    # plt.savefig(f'compare_GP_mono_true.png')
    # plt.savefig(f'compare_GP_mono_mono.pdf')
    # plt.savefig(f'compare_GP_mono_mono.png')
    plt.savefig(f'compare_GP_nomat.pdf')
    plt.savefig(f'compare_GP_nomat.png')
    # plt.savefig(f'compare_GP_mono.pdf')
    # plt.savefig(f'compare_GP_mono.png')
    plt.close()


# Same but instead in 3 subplot, everywhere the True lines are plotted, with one other line per subplot. All same x and y limits. 0 horizontal distance between plots

if plot_type == 'separate':
    num_plots = 3  # 4
    # Plotting
    fig, ax = plt.subplots(1,num_plots,figsize=(.5+2.5*num_plots,2.5), sharex=True, sharey=True)
    # Plot true in all subplots
    for i in range(num_plots):
        ax[i].plot(np.append([0], eps_mono[:, 0]), np.append([0], sig_mono[:, 0]), 'k', alpha=0.4, linestyle=linestyles[0],label='True')
        for j in range(components - 1):
            ax[i].plot(np.append([0], eps_mono[:, j + 1]), np.append([0], sig_mono[:, j + 1]), 'k', alpha=0.4,linestyle=linestyles[0])

    # ax[0].plot(np.append([0], eps_mono[:, 0]), np.append([0], pred_mono[:, 0]), colors[0], linestyle=linestyles[1], label='Mono')
    # ax[1].plot(np.append([0], eps_gp[:, 0]), np.append([0], pred_gp[:, 0]), colors[1], linestyle=linestyles[2], label='GP')
    # ax[2].plot(np.append([0], eps_mono[:, 0]), np.append([0], pred_noepspeq[:, 0]), colors[2], linestyle=linestyles[3], label='No Epspeq')
    # ax[3].plot(np.append([0], eps_mono[:, 0]), np.append([0], pred_nomat[:, 0]), colors[3], linestyle=linestyles[4], label='No Mat')
    ax[0].plot(np.append([0], eps_mono[:, 0]), np.append([0], pred_noepspeq[:, 0]), colors[2], linestyle=linestyles[3], label=r'No $\varepsilon^p_{eq}$')
    ax[1].plot(np.append([0], eps_mono[:, 0]), np.append([0], pred_nomat[:, 0]), colors[3], linestyle=linestyles[4], label='No Material')
    ax[2].plot(np.append([0], eps_mono[:, 0]), np.append([0], pred_fullstrain[:, 0]), colors[4], linestyle=linestyles[5], label='Strain-based')

    for j in range(components-1):
        # ax[0].plot(np.append([0], eps_mono[:, j+1]), np.append([0], pred_mono[:, j+1]), colors[0], linestyle=linestyles[1])
        # ax[1].plot(np.append([0], eps_gp[:, j+1]), np.append([0], pred_gp[:, j+1]), colors[1], linestyle=linestyles[2])
        # ax[2].plot(np.append([0], eps_mono[:, j+1]), np.append([0], pred_noepspeq[:, j+1]), colors[2], linestyle=linestyles[3])
        # ax[3].plot(np.append([0], eps_mono[:, j+1]), np.append([0], pred_nomat[:, j+1]), colors[3], linestyle=linestyles[4])
        ax[0].plot(np.append([0], eps_mono[:, j+1]), np.append([0], pred_noepspeq[:, j+1]), colors[2], linestyle=linestyles[3])
        ax[1].plot(np.append([0], eps_mono[:, j+1]), np.append([0], pred_nomat[:, j+1]), colors[3], linestyle=linestyles[4])
        ax[2].plot(np.append([0], eps_mono[:, j+1]), np.append([0], pred_fullstrain[:, j+1]), colors[4], linestyle=linestyles[5])

    for i in range(num_plots):
        ax[i].legend(loc='lower right')
        ax[i].set_xlabel(r'$\varepsilon^{\Omega}$')
    ax[0].set_ylim([-18.5, 11.5])
    ax[0].set_xlim([-.052, 0.053])

        # remove minor ticks:
        # ax[i].xaxis.set_tick_params(which='minor', bottom=False, top=False)
        # ax[i].yaxis.set_tick_params(which='minor', left=False, right=False)
    # Remove xticks from all
    # for i in range(3):
    #     ax[i].set_xticks([])

    # remove horizontal space between axes
    plt.subplots_adjust(wspace=0, hspace=0)

    ax[0].set_ylabel(r'$\sigma^{\Omega}$')
    plt.tight_layout()
    plt.savefig(f'compare_GP_nomat_subplots_{num_plots}.pdf')
    plt.savefig(f'compare_GP_nomat_subplots_{num_plots}.png')
    plt.close()
