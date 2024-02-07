"""
Plots to make barcharts for comparisons:
- Monotonic vs GP
- No equivalent plastic strain - No material model - Full strain based

Plots stored in: "../experiments/2_unloading/barplot/GPvs..."
"""

import pandas
from matplotlib import pyplot as plt
plt.style.use(['science', 'bright'])
from matplotlib import rc
rc('text', usetex=True)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

"""
Plotting GP vs Monotonic on the unloading_4 dataset.
Updated, model 58_376 GP vs 69_583 Mono.
"""

exp_folder = '../experiments/2_unloading/'

# Load data GP. take average over 2
gp_folder = f"{exp_folder}58_xi_376/test_loss.txt"
losses = pandas.read_csv(gp_folder, sep=',', header=None)
gp_folder2 = f"{exp_folder}58_xi_396/test_loss.txt"
losses2 = pandas.read_csv(gp_folder2, sep=',', header=None)
loss = (losses.iloc[1, 0] + losses2.iloc[1, 0]) / 2
field_strain_loss = (losses.iloc[1, 1] + losses2.iloc[1, 1]) / 2
field_stress_loss = (losses.iloc[1, 2] + losses2.iloc[1, 2]) / 2
hom_stress_loss = (losses.iloc[1, 3] + losses2.iloc[1, 3]) / 2


# Mono data: take average of 2.
mono_folder = f"{exp_folder}69_hyper_noGP2_v2/583_1005/test_loss.txt"
mono_losses = pandas.read_csv(mono_folder, sep=',', header=None)
mono_folder2 = f"{exp_folder}69_hyper_noGP2_v2/582/test_loss.txt"
mono_losses2 = pandas.read_csv(mono_folder2, sep=',', header=None)
mono_loss = (mono_losses.iloc[1, 0] + mono_losses2.iloc[1, 0]) / 2
mono_field_strain_loss = (mono_losses.iloc[1, 1] + mono_losses2.iloc[1, 1]) / 2
mono_field_stress_loss = (mono_losses.iloc[1, 2] + mono_losses2.iloc[1, 2]) / 2
mono_hom_stress_loss = (mono_losses.iloc[1, 3] + mono_losses2.iloc[1, 3]) / 2

# Barchart
invert_colors = [colors[1], colors[0]]
fig, ax = plt.subplots(1,3,figsize=(3,1.5), sharey=True)
ax[0].barh(['GP', 'Mono'], [field_strain_loss, mono_field_strain_loss], color=invert_colors)
ax[1].barh(['GP', 'Mono'], [field_stress_loss, mono_field_stress_loss], color=invert_colors)
ax[2].barh(['GP', 'Mono'], [hom_stress_loss, mono_hom_stress_loss], color=invert_colors)

# Print the quantity inside each bar
for i in range(3):
    for j in range(2):
        if i == 0:
            text = ax[i].text(0, j, f"{ax[i].patches[j].get_width():.4f}", ha='left', va='center', color='white')
        else:
            text = ax[i].text(0, j, f"{ax[i].patches[j].get_width():.2f}", ha='left', va='center', color='white')

ax[0].set_xlabel(r'$\mathcal{L} \varepsilon^{\omega}$ $[-]$')
ax[1].set_xlabel(r'$\mathcal{L} \sigma^{\omega}$ $[MPa]$')
ax[2].set_xlabel(r'$\mathcal{L} \sigma^{\Omega}$ $[MPa]$')
for i in range(3):
    ax[i].xaxis.set_label_position('top')
    ax[i].xaxis.tick_top()
    # Remove all y ticks
    ax[i].yaxis.set_tick_params(which='both', bottom=False, top=False)
# ax[i].set_xlim([0.0, 0.021])

plt.savefig(f'{exp_folder}barplot/GPvsMono.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
plt.savefig(f'{exp_folder}barplot/GPvsMono.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')

"""
Plotting GP vs Monotonic vs noMat vs noEpspeq on the unloading_4 dataset.
Updated, model 58_376 GP vs 69_583 Mono vs 68_570 vs 64_480
"""

exp_folder = '../experiments/2_unloading/'

# Load data GP. take average over 2
gp_folder = f"{exp_folder}58_xi_376/test_loss.txt"
losses = pandas.read_csv(gp_folder, sep=',', header=None)
gp_folder2 = f"{exp_folder}58_xi_396/test_loss.txt"
losses2 = pandas.read_csv(gp_folder2, sep=',', header=None)
loss = (losses.iloc[1, 0] + losses2.iloc[1, 0]) / 2
field_strain_loss = (losses.iloc[1, 1] + losses2.iloc[1, 1]) / 2
field_stress_loss = (losses.iloc[1, 2] + losses2.iloc[1, 2]) / 2
hom_stress_loss = (losses.iloc[1, 3] + losses2.iloc[1, 3]) / 2

# noEpspeq data:  take average of 2.
noEpspeq_folder = f"{exp_folder}68_noepspeq_rep_570/test_loss.txt"
noEpspeq_losses = pandas.read_csv(noEpspeq_folder, sep=',', header=None)
noEpspeq_folder_2 = f"{exp_folder}68_noepspeq_rep_571/test_loss.txt"
noEpspeq_losses_2 = pandas.read_csv(noEpspeq_folder_2, sep=',', header=None)
noEpspeq_loss = (noEpspeq_losses.iloc[1, 0] + noEpspeq_losses_2.iloc[1, 0]) / 2
noEpspeq_field_strain_loss = (noEpspeq_losses.iloc[1, 1] + noEpspeq_losses_2.iloc[1, 1]) / 2
noEpspeq_field_stress_loss = (noEpspeq_losses.iloc[1, 2] + noEpspeq_losses_2.iloc[1, 2]) / 2
noEpspeq_hom_stress_loss = (noEpspeq_losses.iloc[1, 3] + noEpspeq_losses_2.iloc[1, 3]) / 2

# noMat data: take average of 2.
noMat_folder = f"{exp_folder}64_noMat_rep_480/test_loss.txt"
noMat_losses = pandas.read_csv(noMat_folder, sep=',', header=None)
noMat_folder_2 = f"{exp_folder}64_noMat_rep_502/test_loss.txt"
noMat_losses_2 = pandas.read_csv(noMat_folder_2, sep=',', header=None)
noMat_loss = (noMat_losses.iloc[1, 0] + noMat_losses_2.iloc[1, 0]) / 2
noMat_field_strain_loss = (noMat_losses.iloc[1, 1] + noMat_losses_2.iloc[1, 1]) / 2
noMat_field_stress_loss = (noMat_losses.iloc[1, 2] + noMat_losses_2.iloc[1, 2]) / 2
noMat_hom_stress_loss = (noMat_losses.iloc[1, 3] + noMat_losses_2.iloc[1, 3]) / 2

# fullStrain data: take average of 2.
fullStrain_folder = f"{exp_folder}68_noepspeq_rep_572/test_loss.txt"
fullStrain_losses = pandas.read_csv(fullStrain_folder, sep=',', header=None)
fullStrain_folder_2 = f"{exp_folder}68_noepspeq_rep_573/test_loss.txt"
fullStrain_losses_2 = pandas.read_csv(fullStrain_folder_2, sep=',', header=None)
fullStrain_loss = (fullStrain_losses.iloc[1, 0] + fullStrain_losses_2.iloc[1, 0]) / 2
fullStrain_field_strain_loss = (fullStrain_losses.iloc[1, 1] + fullStrain_losses_2.iloc[1, 1]) / 2
fullStrain_field_stress_loss = (fullStrain_losses.iloc[1, 2] + fullStrain_losses_2.iloc[1, 2]) / 2
fullStrain_hom_stress_loss = (fullStrain_losses.iloc[1, 3] + fullStrain_losses_2.iloc[1, 3]) / 2

# Removed mono, added FullStrain
fig, ax = plt.subplots(1,3,figsize=(3,2), sharey=True)
invert_colors = [colors[4], colors[3], colors[2], colors[1]]
names = ['Strain-based', 'No material', r'No $\varepsilon^p_{eq}$', 'Default']
ax[0].barh(names, [fullStrain_field_strain_loss, noMat_field_strain_loss, noEpspeq_field_strain_loss, field_strain_loss], color=invert_colors)
ax[1].barh(names, [fullStrain_field_stress_loss, noMat_field_stress_loss, noEpspeq_field_stress_loss, field_stress_loss], color=invert_colors)
ax[2].barh(names, [fullStrain_hom_stress_loss, noMat_hom_stress_loss, noEpspeq_hom_stress_loss, hom_stress_loss], color=invert_colors)

# Print the quantity inside each bar
for i in range(3):
    for j in range(4):
        if i == 0:
            text = ax[i].text(0, j, f"{ax[i].patches[j].get_width():.4f}", ha='left', va='center', color='white')
        else:
            text = ax[i].text(0, j, f"{ax[i].patches[j].get_width():.2f}", ha='left', va='center', color='white')


ax[0].set_xlabel(r'$\mathcal{L} \varepsilon^{\omega}$ $[-]$')
ax[0].xaxis.set_label_coords(.05, 1.15)
ax[1].set_xlabel(r'$\mathcal{L} \sigma^{\omega}$ $[MPa]$')
ax[2].set_xlabel(r'$\mathcal{L} \sigma^{\Omega}$ $[MPa]$')
for i in range(3):
    ax[i].xaxis.set_label_position('top')
    ax[i].xaxis.tick_top()
    # Remove all y ticks
    ax[i].yaxis.set_tick_params(which='both', bottom=False, top=False)

plt.savefig(f'{exp_folder}barplot/GPvsNoMatvsNoEspepqvsFullStrain.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
plt.savefig(f'{exp_folder}barplot/GPvsNoMatvsNoEspepqvsFullStrain.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')


