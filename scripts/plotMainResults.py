"""
Code used for plotting results of experiments
Each section should be used separately, and can easily be tweaked to create all the other result plots.

Numbers of experiments (60, 62, .. etc) are used internally to track hyperparameter configurations.
Each experiment has the ConfigGNN file used to train the model copied in its folder.
These configuration files can be used to understand how the model was trained exactly.
"""

import os
import numpy as np
import pandas
from readConfig import read_config, get_batch_settings

from matplotlib import pyplot as plt
plt.style.use(['science', 'bright'])
from matplotlib import rc
rc('text', usetex=True)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

"""
Learning curve for the various material model variations.
(By default, results saved in "../experiments/10_testquantity/68_noepspeq/trainsize_seploss_combined_box.pdf")
"""

skip_results = True
plot_60 = True
plot_64 = True
plot_68 = True
plot_68_fullstrain = True

main_folder_60 = '../experiments/10_testquantity/60_learncurve/'
main_folder_64 = '../experiments/10_testquantity/64_noMat/'
main_folder_68 = '../experiments/10_testquantity/68_noepspeq/'
main_folder_68_fullstrain = '../experiments/10_testquantity/68_fullstrain/'

save_folder = main_folder_60
if plot_64:
    save_folder = main_folder_64
if plot_68:
    save_folder = main_folder_68
extra_name = ""
if plot_60 and plot_64:
    extra_name = "_combined"

x_arr = []
loss_arr = []
field_strain_losses = []
field_stress_losses = []
hom_stress_losses = []

for folder in os.listdir(main_folder_60):
    if folder[:4] == 'skip' and skip_results or folder[:4] == 'slurm' or folder[-3:] == '.py' or folder[-4:] == '.pdf' or folder[-4:] == '.png' or folder[:6] == 'LayerN' or folder[:4] == 'fail' or folder[:6] == 'batch4':
        continue

    settings_file = f"{main_folder_60}{folder}/ConfigGNN"
    settings = read_config(settings_file)
    losses = pandas.read_csv(f"{main_folder_60}{folder}/test_loss.txt", sep=',', header=None)

    loss_arr.append(losses.iloc[1, 0])
    field_strain_losses.append(losses.iloc[1, 1])
    field_stress_losses.append(losses.iloc[1, 2])
    hom_stress_losses.append(losses.iloc[1, 3])
    x_arr.append(settings['trainsize'])

x_arr_64 = []
loss_arr_64 = []
field_strain_losses_64 = []
field_stress_losses_64 = []
hom_stress_losses_64 = []

for folder in os.listdir(main_folder_64):
    if folder[:4] == 'skip' and skip_results or folder[:4] == 'slurm' or folder[-3:] == '.py' or folder[-4:] == '.pdf' or folder[-4:] == '.png' or folder[:6] == 'LayerN' or folder[:4] == 'fail' or folder[:6] == 'batch4':
        continue

    settings_file = f"{main_folder_64}{folder}/ConfigGNN"
    settings = read_config(settings_file)
    losses = pandas.read_csv(f"{main_folder_64}{folder}/test_loss.txt", sep=',', header=None)

    loss_arr_64.append(losses.iloc[1, 0])
    field_strain_losses_64.append(losses.iloc[1, 1])
    field_stress_losses_64.append(losses.iloc[1, 2])
    hom_stress_losses_64.append(losses.iloc[1, 3])
    x_arr_64.append(settings['trainsize'])


x_arr_68 = []
loss_arr_68 = []
field_strain_losses_68 = []
field_stress_losses_68 = []
hom_stress_losses_68 = []

for folder in os.listdir(main_folder_68):
    if folder[:4] == 'skip' and skip_results or folder[:4] == 'slurm' or folder[-3:] == '.py' or folder[-4:] == '.pdf' or folder[-4:] == '.png' or folder[:6] == 'LayerN' or folder[:4] == 'fail' or folder[:6] == 'batch4':
        continue

    settings_file = f"{main_folder_68}{folder}/ConfigGNN"
    settings = read_config(settings_file)
    losses = pandas.read_csv(f"{main_folder_68}{folder}/test_loss.txt", sep=',', header=None)

    loss_arr_68.append(losses.iloc[1, 0])
    field_strain_losses_68.append(losses.iloc[1, 1])
    field_stress_losses_68.append(losses.iloc[1, 2])
    hom_stress_losses_68.append(losses.iloc[1, 3])
    x_arr_68.append(settings['trainsize'])


x_arr_68_fullstrain = []
loss_arr_68_fullstrain = []
field_strain_losses_68_fullstrain = []
field_stress_losses_68_fullstrain = []
hom_stress_losses_68_fullstrain = []

for folder in os.listdir(main_folder_68_fullstrain):
    if folder[:4] == 'skip' and skip_results or folder[:4] == 'slurm' or folder[-3:] == '.py' or folder[-4:] == '.pdf' or folder[-4:] == '.png' or folder[:6] == 'LayerN' or folder[:4] == 'fail' or folder[:6] == 'batch4':
        continue

    settings_file = f"{main_folder_68_fullstrain}{folder}/ConfigGNN"
    settings = read_config(settings_file)
    losses = pandas.read_csv(f"{main_folder_68_fullstrain}{folder}/test_loss.txt", sep=',', header=None)

    loss_arr_68_fullstrain.append(losses.iloc[1, 0])
    field_strain_losses_68_fullstrain.append(losses.iloc[1, 1])
    field_stress_losses_68_fullstrain.append(losses.iloc[1, 2])
    hom_stress_losses_68_fullstrain.append(losses.iloc[1, 3])
    x_arr_68_fullstrain.append(settings['trainsize'])


# Standard plot
plt.figure(figsize=(3, 2.5))

if plot_60:
    plt.scatter(x_arr, loss_arr, label=r'$\sigma = f(\varepsilon(x))$')
if plot_68:
    plt.scatter(x_arr_68, loss_arr_68, label='No ' + r'$\varepsilon^{p}_{eq}$')
if plot_64:
    plt.scatter(x_arr_64, loss_arr_64, label='$\sigma = GNN(x)$')
if plot_68:
    plt.scatter(x_arr_68, loss_arr_68, label='$\sigma = GNN(x)$')

xticks = np.unique(x_arr)
plt.xticks(xticks)
plt.xlabel('Training samples', fontsize=12)
plt.minorticks_off()
plt.ylabel(r'$\mathcal{L}$', fontsize=12)
plt.legend()
print("saving figure...")
plt.savefig(f'{save_folder}/trainsize{extra_name}.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
plt.savefig(f'{save_folder}/trainsize{extra_name}.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')

# Set colors for consistency with other plots:
colors = [colors[1], colors[3], colors[2], colors[4]]

# Per loss
fig, ax = plt.subplots(2,2, figsize=(6, 4))
if plot_60:
    plot1 = ax[0,0].scatter(x_arr, loss_arr, color=colors[0])
    plot1 = ax[1,0].scatter(x_arr, field_strain_losses, color=colors[0], label=r'A: Base model')
    plot2 = ax[0,1].scatter(x_arr, field_stress_losses, color=colors[0])
    # plot3 = ax[3].scatter(x_arr, hom_stress_losses, color=colors[0])

if plot_68:
    plot1 = ax[0,0].scatter(x_arr_68, loss_arr_68, marker='*', color=colors[2])
    plot1 = ax[1,0].scatter(x_arr_68, field_strain_losses_68, marker='*', color=colors[2], label='B: No ' + r'$\varepsilon^{p}_{eq}$')
    plot2 = ax[0,1].scatter(x_arr_68, field_stress_losses_68, marker='*', color=colors[2])
    # plot3 = ax[3].scatter(x_arr_68, hom_stress_losses_68, marker='*', color=colors[2])

if plot_64:
    plot1 = ax[0,0].scatter(x_arr_64, loss_arr_64, marker='x', color=colors[1])
    plot1 = ax[1,0].scatter(x_arr_64, field_strain_losses_64, marker='x', color=colors[1], label='C: No material')
    plot2 = ax[0,1].scatter(x_arr_64, field_stress_losses_64, marker='x', color=colors[1])
#     plot3 = ax[3].scatter(x_arr_64, hom_stress_losses_64, marker='x', color=colors[1])

if plot_68_fullstrain:
    plot1 = ax[0,0].scatter(x_arr_68_fullstrain, loss_arr_68_fullstrain, marker='+', color=colors[3])
    plot1 = ax[1,0].scatter(x_arr_68_fullstrain, field_strain_losses_68_fullstrain, marker='+', color=colors[3], label='D: Strain-based')
    plot2 = ax[0,1].scatter(x_arr_68_fullstrain, field_stress_losses_68_fullstrain, marker='+', color=colors[3])
    # plot3 = ax[3].scatter(x_arr_68_fullstrain, hom_stress_losses_68_fullstrain, marker='*', color=colors[2])


# Turn off everything for the bottom right plot
ax[1,1].axis('off')
# Increase vertical space
plt.subplots_adjust(hspace=0.3)

# logscale
# ax[0,0].set_xscale('log')
# ax[1,0].set_xscale('log')
# ax[0,1].set_xscale('log')

ax[0,0].minorticks_off()
ax[0,0].set_xticks(xticks)
ax[0,0].set_xlabel('N $[-]$', fontsize=12)
ax[0,0].xaxis.set_label_coords(.75, -0.05)
ax[1,0].minorticks_off()
ax[1,0].set_xticks(xticks)
ax[1,0].set_xlabel('Training samples N $[-]$', fontsize=12)
ax[0,1].minorticks_off()
ax[0,1].set_xticks(xticks)
ax[0,1].set_xlabel('Training samples N $[-]$', fontsize=12)
# ax[0,1].xaxis.set_label_coords(0.7, -0.05)

ax[0,0].set_ylabel(r'$\mathcal{L}$', fontsize=12)
ax[1,0].set_ylabel(r'$\mathcal{L} \varepsilon^{\omega}$ $[-]$', fontsize=12)
ax[0,1].set_ylabel(r'$\mathcal{L} \sigma^{\omega}$ $[MPa]$', fontsize=12)
# ax[3].set_ylabel(r'$\mathcal{L} \sigma^{\Omega}$', fontsize=14)

# Rotate the xticks 30 degrees
for tick in ax[0,0].get_xticklabels():
    tick.set_rotation(30)
for tick in ax[1,0].get_xticklabels():
    tick.set_rotation(30)
for tick in ax[0,1].get_xticklabels():
    tick.set_rotation(30)

# move legend location outside of plot to bottom right corner not used
ax[1,0].legend(loc='lower right', bbox_to_anchor=(2.0, -0.1), fontsize=12)

print("saving figure...")
plt.savefig(f'{save_folder}/trainsize_seploss{extra_name}_box.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
plt.savefig(f'{save_folder}/trainsize_seploss{extra_name}_box.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')

"""
Results 61 & 63 & 66 & 69 number of Message Passing Layers
Hyperparameter plots for various training configurations.
"""
skip_results = True
# main_folder = '../experiments/10_testquantity/61_MPLs/'
# main_folder = '../experiments/10_testquantity/63_hyper_noGP/MPL/'
# main_folder = '../experiments/10_testquantity/66_hyper_noGP2/MPL/'
main_folder = '../experiments/10_testquantity/69_hyper_noGP2_v2/MPL/'

MPL_arr = []
loss_arr = []   # 2 = number of runs combined in data
field_strain_losses = []   # 2 = number of runs combined in data
field_stress_losses = []
hom_stress_losses = []   # 2 = number of runs combined in data

for folder in os.listdir(main_folder):
    if folder[:4] == 'skip' and skip_results or folder == 'slurm' or folder[-3:] == '.py' or folder[-4:] == '.pdf' or folder[-4:] == '.png' or folder[:6] == 'LayerN' or folder[:4] == 'fail' or folder[:6] == 'batch4':
        continue
    scaling_factor = 1
    if folder == 'MPL_7_434' or folder == 'MPL_3_431' or folder == 'MPL_1_438':
        scaling_factor = np.sqrt(2000)
    settings_file = f"{main_folder}{folder}/ConfigGNN"
    print(settings_file)
    settings = read_config(settings_file)
    losses = pandas.read_csv(f"{main_folder}{folder}/test_loss.txt", sep=',', header=None)

    loss_arr.append(losses.iloc[1, 0] * scaling_factor)
    field_strain_losses.append(losses.iloc[1, 1])
    field_stress_losses.append(losses.iloc[1, 2])
    hom_stress_losses.append(losses.iloc[1, 3])
    MPL_arr.append(settings['layers_arr'][0])

# Standard plot
plt.figure(figsize=(3, 2.5))
plt.scatter(MPL_arr, loss_arr)
plt.xticks(np.unique(MPL_arr))
plt.xlabel('MPLs', fontsize=14)
plt.minorticks_off()
plt.gca().yaxis.set_label_coords(-0.18, 0.5)
plt.ylabel(r'$\mathcal{L}$', fontsize=14)
print("saving figure...")
plt.savefig(f'{main_folder}/MPLs.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
plt.savefig(f'{main_folder}/MPLs.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')

# Per loss
fig, ax = plt.subplots(4,1, figsize=(3, 4.5), sharex=True)
plot1 = ax[0].scatter(MPL_arr, loss_arr, color=colors[0])
plot1 = ax[1].scatter(MPL_arr, field_strain_losses, color=colors[0])
plot2 = ax[2].scatter(MPL_arr, field_stress_losses, color=colors[0])
plot3 = ax[3].scatter(MPL_arr, hom_stress_losses, color=colors[0])
ax[0].minorticks_off()
ax[1].minorticks_off()
ax[2].minorticks_off()
ax[3].minorticks_off()
ax[3].set_xticks(np.unique(MPL_arr))
ax[3].set_xlabel('MPLs', fontsize=14)
ax[0].set_ylabel(r'$\mathcal{L}$', fontsize=14)
ax[1].set_ylabel(r'$\mathcal{L} \varepsilon^{\omega}$', fontsize=14)
ax[2].set_ylabel(r'$\mathcal{L} \sigma^{\omega}$', fontsize=14)
ax[3].set_ylabel(r'$\mathcal{L} \sigma^{\Omega}$', fontsize=14)


print("saving figure...")
plt.savefig(f'{main_folder}/MPLs_seploss.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
plt.savefig(f'{main_folder}/MPLs_seploss.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')

"""
Results 62 & 63 & 66 & 69 dropout
Hyperparameter plots for various training configurations.
"""
skip_results = True
# main_folder = '../experiments/10_testquantity/62_dropout/'
# main_folder = '../experiments/10_testquantity/63_hyper_noGP/dropout/'
# main_folder = '../experiments/10_testquantity/66_hyper_noGP2/dropout/'
main_folder = '../experiments/10_testquantity/69_hyper_noGP2_v2/dropout/'

x_arr = []
loss_arr = []   # 2 = number of runs combined in data
field_strain_losses = []   # 2 = number of runs combined in data
field_stress_losses = []
hom_stress_losses = []   # 2 = number of runs combined in data

for folder in os.listdir(main_folder):
    if folder[:4] == 'skip' and skip_results or folder == 'slurm' or folder[-3:] == '.py' or folder[-4:] == '.pdf' or folder[-4:] == '.png' or folder[:6] == 'LayerN' or folder[:4] == 'fail' or folder[:6] == 'batch4':
        continue

    settings_file = f"{main_folder}{folder}/ConfigGNN"
    settings = read_config(settings_file)
    losses = pandas.read_csv(f"{main_folder}{folder}/test_loss.txt", sep=',', header=None)

    loss_arr.append(losses.iloc[1, 0])
    field_strain_losses.append(losses.iloc[1, 1])
    field_stress_losses.append(losses.iloc[1, 2])
    hom_stress_losses.append(losses.iloc[1, 3])
    x_arr.append(settings['dropout'])

# Standard plot
plt.figure(figsize=(3, 2.5))
plt.scatter(x_arr, loss_arr)
plt.xlabel('Dropout rate', fontsize=14)
plt.ylabel(r'$\mathcal{L}$', fontsize=14)
plt.minorticks_off()
# Set the yaxis label coords to -0.18, 0.5:
plt.gca().yaxis.set_label_coords(-0.18, 0.5)

print("saving figure...")
plt.savefig(f'{main_folder}/dropout.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
plt.savefig(f'{main_folder}/dropout.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')

# Per loss
fig, ax = plt.subplots(4,1, figsize=(3, 4.5), sharex=True)
plot1 = ax[0].scatter(x_arr, loss_arr, color=colors[0])
plot1 = ax[1].scatter(x_arr, field_strain_losses, color=colors[0])
plot2 = ax[2].scatter(x_arr, field_stress_losses, color=colors[0])
plot3 = ax[3].scatter(x_arr, hom_stress_losses, color=colors[0])
ax[0].minorticks_off()
ax[1].minorticks_off()
ax[2].minorticks_off()
ax[3].minorticks_off()
ax[3].set_xticks(x_arr)
ax[3].set_xlabel('Dropout rate', fontsize=14)
ax[0].set_ylabel(r'$\mathcal{L}$', fontsize=14)
ax[1].set_ylabel(r'$\mathcal{L} \varepsilon^{\omega}$', fontsize=14)
ax[2].set_ylabel(r'$\mathcal{L} \sigma^{\omega}$', fontsize=14)
ax[3].set_ylabel(r'$\mathcal{L} \sigma^{\Omega}$', fontsize=14)

ax[0].yaxis.set_label_coords(-0.18, 0.5)
ax[1].yaxis.set_label_coords(-0.18, 0.5)
ax[2].yaxis.set_label_coords(-0.18, 0.5)
ax[3].yaxis.set_label_coords(-0.18, 0.5)

print("saving figure...")
plt.savefig(f'{main_folder}/dropout_seploss.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
plt.savefig(f'{main_folder}/dropout_seploss.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')

"""
Results 63 & 66 & 58 xi single plots
Hyperparameter plots for various training configurations.
"""
skip_results = True
# main_folder = '../experiments/10_testquantity/58_xi_field_1-9/'
# main_folder = '../experiments/10_testquantity/63_hyper_noGP/xi/'
# main_folder = '../experiments/10_testquantity/66_hyper_noGP2/xi/'
main_folder = '../experiments/10_testquantity/69_hyper_noGP2_v2/xi/'

x_arr = []
loss_arr = []
field_strain_losses = []
field_stress_losses = []
hom_stress_losses = []

for folder in os.listdir(main_folder):
    if folder[:4] == 'skip' and skip_results or folder == 'slurm' or folder[-3:] == '.py' or folder[-4:] == '.pdf' or folder[-4:] == '.png' or folder[:6] == 'LayerN' or folder[:4] == 'fail' or folder[:6] == 'batch4':
        continue


    settings_file = f"{main_folder}{folder}/ConfigGNN"
    settings = read_config(settings_file)
    losses = pandas.read_csv(f"{main_folder}{folder}/test_loss.txt", sep=',', header=None)

    loss_arr.append(losses.iloc[1, 0])
    field_strain_losses.append(losses.iloc[1, 1])
    field_stress_losses.append(losses.iloc[1, 2])
    hom_stress_losses.append(losses.iloc[1, 3])
    x_arr.append(settings['xi_stressfield'])

# Standard plot
plt.figure(figsize=(3, 2.5))
plt.scatter(x_arr, loss_arr)
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel(r'$\mathcal{L}$', fontsize=14)
plt.minorticks_off()
print("saving figure...")
plt.savefig(f'{main_folder}/xi.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
plt.savefig(f'{main_folder}/xi.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')

# Per loss
fig, ax = plt.subplots(2,1, figsize=(3, 2.5), sharex=True)
# plot1 = ax[0].scatter(x_arr, loss_arr, color=colors[0])
plot1 = ax[0].scatter(x_arr, field_strain_losses, color=colors[0])
plot2 = ax[1].scatter(x_arr, field_stress_losses, color=colors[0])
# plot3 = ax[3].scatter(x_arr, hom_stress_losses, color=colors[0])
ax[0].minorticks_off()
ax[1].minorticks_off()
# ax[2].minorticks_off()
# ax[3].minorticks_off()
ax[1].set_xticks(np.unique(x_arr))
ax[1].set_xlabel(r'$\xi$', fontsize=14)
# ax[0].set_ylabel(r'$\mathcal{L}$', fontsize=14)
ax[0].set_ylabel(r'$\mathcal{L} \varepsilon^{\omega}$ $[-]$', fontsize=14)
ax[1].set_ylabel(r'$\mathcal{L} \sigma^{\omega}$ $[MPa]$', fontsize=14)
ax[0].yaxis.set_label_coords(-0.18, 0.5)
ax[1].yaxis.set_label_coords(-0.18, 0.5)
# ax[3].set_ylabel(r'$\mathcal{L} \sigma^{\Omega}$', fontsize=14)

print("saving figure...")
plt.savefig(f'{main_folder}/xi_sep2.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
plt.savefig(f'{main_folder}/xi_sep2.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')

"""
Experiment 4: time extrapolatoin. Combine baseline with predictions
"""
main_folder = '../experiments/4_time_extrap/GP_376_norm'
nomat_folder = '../experiments/4_time_extrap/64_noMat_norm'

plot_scatter = False
plot_noMat = True
plot_macro_base = False
plot_baseline = False
num_samples = 2000
train_steps = 25
num_steps = 50
plot_folder = main_folder + '/plots'

txt_files = ['eps_t_errors', 'eps_max_t_errors', 'sig_t_errors', 'sig_max_t_errors', 'homsig_t_errors', 'homsig_max_t_errors']

# Read data
for file in txt_files:
    data = np.loadtxt(f'{main_folder}/{file}.txt')
    if file == 'eps_t_errors':
        mean_eps_field = data
    elif file == 'eps_max_t_errors':
        max_eps_field = data
    elif file == 'sig_t_errors':
        mean_sig_field = data
    elif file == 'sig_max_t_errors':
        max_sig_field = data
    elif file == 'homsig_t_errors':
        mean_homsig = data
    elif file == 'homsig_max_t_errors':
        max_homsig = data

# Read no material data (if None -> not plotted)
mean_eps_field_nomat = None
max_eps_field_nomat = None
mean_sig_field_nomat = None
max_sig_field_nomat = None
mean_homsig_nomat = None
max_homsig_nomat = None
if plot_noMat:
    for file in txt_files:
        data = np.loadtxt(f'{nomat_folder}/{file}.txt')
        if file == 'eps_t_errors':
            mean_eps_field_nomat = data
        elif file == 'eps_max_t_errors':
            max_eps_field_nomat = data
        elif file == 'sig_t_errors':
            mean_sig_field_nomat = data
        elif file == 'sig_max_t_errors':
            max_sig_field_nomat = data
        elif file == 'homsig_t_errors':
            mean_homsig_nomat = data
        elif file == 'homsig_max_t_errors':
            max_homsig_nomat = data

# Read baseline data
base_mean_eps_field = None
base_max_eps_field = None
base_mean_sig_field = None
base_max_sig_field = None
base_mean_homsig = None
base_max_homsig = None
if plot_baseline:
    for file in txt_files:
        data = np.loadtxt(f'{main_folder}_baseline/{file}.txt')
        if file == 'eps_t_errors':
            base_mean_eps_field = data
        elif file == 'eps_max_t_errors':
            base_max_eps_field = data
        elif file == 'sig_t_errors':
            base_mean_sig_field = data
        elif file == 'sig_max_t_errors':
            base_max_sig_field = data
        elif file == 'homsig_t_errors':
            base_mean_homsig = data
        elif file == 'homsig_max_t_errors':
            base_max_homsig = data

# Read baseline average strain
base_macro_mean_eps_field = None
base_macro_max_eps_field = None
if plot_macro_base:
    for file in txt_files:
        data = np.loadtxt(f'{main_folder}_baseline_macro/{file}.txt')
        if file == 'eps_t_errors':
            base_macro_mean_eps_field = data
        elif file == 'eps_max_t_errors':
            base_macro_max_eps_field = data

try:
    os.mkdir(f'{plot_folder}')
except:
    print(f"Folder {plot_folder} already exists")
    pass

def plot_t_error(errors, base_errors, train_steps, type, label, extra_data = None, plot_legend = False, yticks=None, moresteps = None, morestepsdata = None, macro_base = None):  # Type = Average or Max
    mean_time_error = np.mean(errors, axis=0)
    if plot_baseline:
        base_mean = np.mean(base_errors, axis=0)
    if extra_data is not None:
        extra_mean = np.mean(extra_data, axis=0)
        height_plot = max(np.max(mean_time_error), extra_mean.max()) * 1.1
    elif plot_baseline:
        if extra_data is not None:
            extra_mean = np.mean(extra_data, axis=0)
            height_plot = max(np.max(mean_time_error), np.max(base_mean), extra_mean.max()) * 1.1
        else:
            height_plot = max(np.max(mean_time_error), np.max(base_mean)) * 1.1
    else:
        height_plot = np.max(mean_time_error) * 1.1
    if morestepsdata is not None:
        moresteps_mean = np.mean(morestepsdata, axis=0)
        height_plot = max(height_plot, moresteps_mean.max()* 1.1)
    if macro_base is not None:
        macro_base_mean = np.mean(macro_base, axis=0)
        height_plot = max(height_plot, macro_base_mean.max()* 1.1)

    # fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    x_arr = np.arange(len(mean_time_error)) + 1
    x_arr_expanded = np.broadcast_to(x_arr, (errors.shape))

    # Background colors & text
    if not plot_scatter:
        ax.fill_between([0, train_steps], 0, height_plot, color=colors[0], alpha=0.1)
        ax.fill_between([train_steps, num_steps+3], 0, height_plot, color=colors[1], alpha=0.2)
        ax.vlines([25], 0, height_plot, color='k', linewidth=.2)
        ax.text(0.23, 0.03, 'Training', transform = ax.transAxes)
        ax.text(0.5, 0.03, 'Extrapolating', transform = ax.transAxes)

    if plot_baseline:
        ax.plot(np.append([0],x_arr), np.append([0],base_mean), '--', label='0 Baseline', color='k')
        if macro_base is not None:
            ax.plot(np.append([0],x_arr), np.append([0],macro_base_mean), '--', label=r'$\varepsilon^{\Omega}$ Baseline', color='b')

    if plot_scatter:
        ax.scatter(x_arr_expanded[:,:train_steps], errors[:,:train_steps], s=4, color=colors[0], alpha=0.1)  # FEM
        ax.scatter(x_arr_expanded[:, train_steps:], errors[:, train_steps:], s=4, color=colors[1], alpha=0.1)  # FEM

    if extra_data is not None:
        ax.plot(np.append([0],x_arr[:train_steps]), np.append([0],extra_mean[:train_steps]), '-.', label='No material',color=colors[2])
        ax.plot(x_arr[train_steps - 1:], extra_mean[train_steps - 1:], '-.',color=colors[3])

    if morestepsdata is not None:
        ax.plot(np.append([0],x_arr[:moresteps]), np.append([0],moresteps_mean[:moresteps]), ':', label='More steps',color=colors[4])
        ax.plot(x_arr[moresteps - 1:], moresteps_mean[moresteps - 1:], ':',color=colors[5])

    ax.plot(np.append([0],x_arr[:train_steps]), np.append([0],mean_time_error[:train_steps]), label='Base model',color=colors[0])
    ax.plot(x_arr[train_steps - 1:], mean_time_error[train_steps - 1:],color=colors[1])

    # turn of minor xticks
    ax.minorticks_off()
    ax.set_xlim([0,52])
    ax.set_ylim([0,height_plot])
    ax.set_xticks([0, 10, 20, 25, 30, 40, 50])
    if yticks is not None:
        ax.set_yticks(yticks)

    plt.xlabel('Timestep')
    # plt.ylabel(f'{type} ' + r'$ ||\hat{\varepsilon}^{\omega}-\varepsilon^{\omega}||^2 $')   # MSE
    plt.ylabel(label)    # MAE
    ax.yaxis.set_label_coords(-0.19, 0.5)

    if plot_legend:
        if plot_scatter:
            # legend at relative coordinates 0.7, 0.1
            plt.legend(loc='upper left')
        else:
            plt.legend(loc='upper right', bbox_to_anchor=(1.03, 0.52))
    if plot_scatter:
        plt.savefig(f'{plot_folder}/t_{type}_base.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
        plt.savefig(f'{plot_folder}/t_{type}_base.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')
    else:
        plt.savefig(f'{plot_folder}/t_{type}_noscat.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
        plt.savefig(f'{plot_folder}/t_{type}_noscat.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')

# Norm
plot_t_error(mean_eps_field, base_mean_eps_field, train_steps, 'eps_Average',f'Mean ' + r'$ ||\hat{\varepsilon}^{\omega}-\varepsilon^{\omega}|| \;[-]$', extra_data = mean_eps_field_nomat, plot_legend=True, macro_base = base_macro_mean_eps_field )
plot_t_error(max_eps_field, base_max_eps_field, train_steps, 'eps_Max', f'Max ' + r'$ ||\hat{\varepsilon}^{\omega}-\varepsilon^{\omega}|| \;[-]$', extra_data = max_eps_field_nomat, macro_base = base_macro_max_eps_field )
plot_t_error(mean_sig_field, base_mean_sig_field, train_steps, 'sig_Average', f'Mean ' + r'$ ||\hat{\sigma}^{\omega}-\sigma^{\omega}|| \;[MPA]$', extra_data = mean_sig_field_nomat )
plot_t_error(max_sig_field, base_max_sig_field, train_steps, 'sig_Max', f'Max ' + r'$ ||\hat{\sigma}^{\omega}-\sigma^{\omega}|| \;[MPA]$', extra_data = max_sig_field_nomat)
plot_t_error(mean_homsig, base_mean_homsig, train_steps, 'sig_hom', r'$ ||\hat{\sigma}^{\Omega}-\sigma^{\Omega}|| \;[MPA]$', extra_data = mean_homsig_nomat)
