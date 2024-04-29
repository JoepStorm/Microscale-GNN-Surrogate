"""
Utilities file for all plotting functions related to training and the direct results.
- Plotting of losses throughout training
- Plotting of resulting microscopic fields
- Plotting of the homogenized stress-strain curves.
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import rc
rc('text', usetex=True)
plt.style.use(['science', 'bright'])
from PyPDF2 import PdfMerger
import os


"""
Plotting
"""
def plot_losses(loss_arrays, loss_names, settings, dirstring, best_epoch = None, true_val = False, best_val = None):
    """
    Expects the following:
    loss_arrays[0] = training loss
    loss_arrays[1] = validation loss
    if there is at least 1 transfer sample:
        loss_arrays[2] = transfer loss
    """
    plt.figure()
    plt.plot(loss_arrays[0], label=loss_names[0])
    plt.plot( np.arange(0, len(loss_arrays[1]), settings['val_frequency']) ,loss_arrays[1][::settings['val_frequency']], label=loss_names[1])
    height_plot = max(max(loss_arrays[0]), max(loss_arrays[1]))

    if true_val:
        plt.plot(np.arange(0, len(loss_arrays[3]), settings['full_val_frequency']),
                 loss_arrays[3][::settings['full_val_frequency']], label=loss_names[3])
        height_plot = max(height_plot, max(loss_arrays[3]))

    if settings['validationend'] != settings['transferend'] and settings['transfertrain']:
        plt.plot(np.arange(0, len(loss_arrays[2]), settings['transf_frequency']),
                 loss_arrays[2][::settings['transf_frequency']], label='Transfer', color='#AA3377')
        height_plot = max(height_plot, max(loss_arrays[2]))
    if best_epoch is not None:
        plt.vlines(best_epoch, 0, height_plot, color='#66CCEE', linestyles='dashed', label='best model')

    if best_val is not None:
        plt.scatter(best_epoch, best_val, color='#228833', s=8, label='Full step loss')
        height_plot = max(height_plot, best_val*1.1)

    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.xlim(xmin=0)
    plt.legend(fontsize=6, ncol=2)
    plt.ylim(0, height_plot)
    plt.savefig(f"{dirstring}/RMSE.pdf", bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')

def plot_sep_losses(sep_losses_arr, sep_losses_arr_norm, settings, dirstring, best_epoch=None):
    """
    Expects the following:
    loss_arrays[0] = training loss
    loss_arrays[1] = validation loss
    if there is at least 1 transfer sample:
        loss_arrays[2] = transfer loss
    """
    plt.figure()
    epochs = len(sep_losses_arr[0])

    newcurve_1 = plt.plot(np.arange(0, epochs), sep_losses_arr[0], label=f'Train - strainfield')
    newcurve = plt.plot(np.arange(0, epochs), sep_losses_arr[1], '--', label=f'Train - stressfield', color=newcurve_1[0].get_color())
    newcurve_2 = plt.plot(np.arange(0, epochs, settings['val_frequency']), sep_losses_arr[3][::settings['val_frequency']], label=f'Val - strainfield')
    newcurve = plt.plot(np.arange(0, epochs, settings['val_frequency']), sep_losses_arr[4][::settings['val_frequency']], '--', label=f'Val - stressfield', color=newcurve_2[0].get_color())
    if settings['transferend'] != settings['validationend'] and settings['transfertrain']:
        newcurve_3 = plt.plot(np.arange(0, epochs, settings['transf_frequency']), sep_losses_arr[6][::settings['transf_frequency']], label=f'Transf - strainfield', color='#AA3377')
        newcurve = plt.plot(np.arange(0, epochs, settings['transf_frequency']), sep_losses_arr[7][::settings['transf_frequency']], '--', label=f'Transf - stressfield', color=newcurve_3[0].get_color())

    if best_epoch is not None:
        plt.axvline(best_epoch, color='#66CCEE', linestyle='dashed', label='best model')

    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.xlim(xmin=0)
    plt.legend(fontsize=6, loc='upper right', ncol=2)
    plt.title('Full-field - unnormalized')

    plt.ylim(ymin=0,ymax=sep_losses_arr[2][0] * 1.2)
    plt.savefig(f"{dirstring}/RMSE_field_unnorm.pdf", bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')

    # 2nd Plot with the homogenized quantities also in there!
    plt.plot(np.arange(0, epochs), sep_losses_arr[2], '-.', label='Homogenized stress', color=newcurve_1[0].get_color())
    plt.plot(np.arange(0, epochs, settings['val_frequency']), sep_losses_arr[5][::settings['val_frequency']], '-.', color=newcurve_2[0].get_color())
    if settings['transferend'] != settings['validationend'] and settings['transfertrain']:
        plt.plot(np.arange(0, epochs, settings['transf_frequency']), sep_losses_arr[8][::settings['transf_frequency']], '-.', color=newcurve_3[0].get_color())
    plt.ylim(ymax=max(sep_losses_arr[3][0] * 1.2, sep_losses_arr[4][0] * 1.2, sep_losses_arr[5][0] * 1.2))
    plt.title('All - unnormalized')
    plt.legend(fontsize=6, loc='upper right', ncol=2)

    plt.savefig(f"{dirstring}/RMSE_sep.pdf", bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
    
    
    # Normalized plot
    plt.figure()
    newcurve_1 = plt.plot(np.arange(0, epochs), sep_losses_arr_norm[0], label=f'Train - strainfield')
    newcurve = plt.plot(np.arange(0, epochs), sep_losses_arr_norm[1], '--', label=f'Train - stressfield', color=newcurve_1[0].get_color())
    newcurve_2 = plt.plot(np.arange(0, epochs, settings['val_frequency']), sep_losses_arr_norm[3][::settings['val_frequency']], label=f'Val - strainfield')
    newcurve = plt.plot(np.arange(0, epochs, settings['val_frequency']), sep_losses_arr_norm[4][::settings['val_frequency']], '--', label=f'Val - stressfield', color=newcurve_2[0].get_color())
    if settings['xi_stresshom'] != 0.0:
        plt.plot(np.arange(0, epochs), sep_losses_arr_norm[2], '-.', label='Homogenized stress', color=newcurve_1[0].get_color())
        plt.plot(np.arange(0, epochs, settings['val_frequency']), sep_losses_arr_norm[5][::settings['val_frequency']], '-.', color=newcurve_2[0].get_color())
    if settings['transferend'] != settings['validationend'] and settings['transfertrain']:
        newcurve_3 = plt.plot(np.arange(0, epochs, settings['transf_frequency']), sep_losses_arr_norm[6][::settings['transf_frequency']], label=f'Transf - strainfield', color='#AA3377')
        newcurve = plt.plot(np.arange(0, epochs, settings['transf_frequency']), sep_losses_arr_norm[7][::settings['transf_frequency']], '--', label=f'Transf - stressfield', color=newcurve_3[0].get_color())
        if settings['xi_stresshom'] != 0.0:
            plt.plot(np.arange(0, epochs, settings['transf_frequency']), sep_losses_arr_norm[8][::settings['transf_frequency']], '-.', color=newcurve_3[0].get_color())
    
    if best_epoch is not None:
        plt.axvline(best_epoch, color='#66CCEE', linestyle='dashed', label='best model')
    
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.xlim(xmin=0)
    plt.ylim(ymax=max(sep_losses_arr_norm[3][0] * 1.2, sep_losses_arr_norm[4][0] * 1.2, sep_losses_arr_norm[5][0] * 1.2))
    plt.title('All - normalized')
    plt.legend(fontsize=6, loc='upper right', ncol=2)

    plt.savefig(f"{dirstring}/RMSE_norm.pdf", bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')


"""
File to plot strains by colors in the mesh 
"""
def triangle_coords(mesh_node_coords, mesh_elem_nodes, elem):
    new = np.zeros((3,2))
    for k in range(3):
        new[k] = mesh_node_coords[mesh_elem_nodes[elem][k]]
    return new

def color_plot(tri_mesh, colors, str_title, fname, bound_values, colorscheme, label, ipCoords=None, ipSizes=None):

    cmap_options = {
        'jet': cm.jet,
        'bwr': cm.bwr
    }

    cmap = cmap_options[colorscheme]

    fig, ax = plt.subplots()
    norm = mpl.colors.Normalize(vmin=bound_values[0], vmax=bound_values[1])

    for i, cur_triang in enumerate(tri_mesh):
        plt.fill(cur_triang[:, 0], cur_triang[:, 1], c=cmap(norm(colors[i])), edgecolor='black', linewidth=0.01, zorder=-1)

    if ipCoords is not None and ipSizes is not None:
        for i in range(len(ipCoords)):
            plt.scatter(ipCoords[i,0], ipCoords[i,1], c='black', s=ipSizes[i], zorder=1 )
    elif ipCoords is not None:
        plt.scatter(ipCoords[:,0], ipCoords[:,1], c='black', s=2, zorder=1 )

    ax.set_aspect('equal')
    ax.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.title(str_title, fontsize=8) #, loc='right')
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), shrink=0.7)
    cbar.ax.set_title(label)
    plt.savefig(fname + '.pdf', bbox_inches='tight', pad_inches=0.01, dpi=500, format='pdf')
    plt.close('all')

def color_plot_combined(tri_mesh, all_colors, str_titles, fname, bound_values, colorschemes, labels, ipCoords=None, ipSizes=None):
    cmap_options = {
        'jet': cm.jet,
        'bwr': cm.bwr,
        'Spectral': cm.Spectral
    }

    # cmap = cm.jet
    num = len(all_colors)
    fig, axs = plt.subplots(1,num, figsize=(10, 4))

    for i in range(num):
        cmap = cmap_options[colorschemes[i]]
        colors = all_colors[i]

        # Default, 0 isn't right
        norm = mpl.colors.Normalize(vmin=bound_values[i][0], vmax=bound_values[i][1]) #, vcenter=0)

        # Fancy, 0 is correct. However the cmap isnt a linear scale, above and below are scaled differently....
        # norm = mpl.colors.TwoSlopeNorm(vmin=bound_values[i][0], vcenter=0, vmax=bound_values[i][1])

        # Default but [-max, max] to force 0 in middle..
        max_val = max(abs(bound_values[i][0]), abs(bound_values[i][1]))
        norm = mpl.colors.Normalize(vmin=-max_val, vmax=max_val) #, vcenter=0)

        for j, cur_triang in enumerate(tri_mesh):
            axs[i].fill(cur_triang[:, 0], cur_triang[:, 1], c=cmap(norm(colors[j])), edgecolor='black', linewidth=0.01, zorder=-1)
            # axs[i].fill(cur_triang[:, 0], cur_triang[:, 1], c=cmap(norm(colors[j])), edgecolor=cmap(norm(colors[j])), linewidth=0.01, zorder=-1)

        if ipCoords is not None and ipSizes is not None:
            for j in range(len(ipCoords)):
                axs[i].scatter(ipCoords[j,0], ipCoords[j,1], c='black', s=ipSizes[j], zorder=1 )
        elif ipCoords is not None:
            axs[i].scatter(ipCoords[:,0], ipCoords[:,1], c='black', s=2, zorder=1 )

        axs[i].set_aspect('equal')
        axs[i].axis('off')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(str_titles[i], fontsize=8) #, loc='right')
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), shrink=0.5, ax=axs[i])
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.01, dpi=500, format='pdf')
    plt.close('all')

def rve_unroll_visual( dirstring, strains, stresses, epspeqs, norm_stresses, elemStressNorm, m, timesteps, mesh_ipArea_single, macrostrain, homstress, homStressNorm, targets_all_steps, epspeq_all_steps, mesh_node_coords_set, mesh_elem_nodes_set, settings, final_only, extra_title = '', homstress_pred = None):

    # Triangulate mesh
    mesh_node_coords = mesh_node_coords_set[m]
    mesh_elem_nodes = np.array(mesh_elem_nodes_set[m]) - 1
    tri_mesh = np.empty((len(mesh_elem_nodes), 3, 2))
    for i, node in enumerate(mesh_elem_nodes):
        cur_triang = triangle_coords(mesh_node_coords, mesh_elem_nodes, i)
        tri_mesh[i] = cur_triang

    num_nodes = len(mesh_elem_nodes)

    titles = []
    for t in range(timesteps):
        macro_title = r"$\varepsilon^{\Omega} =$" + f"[{macrostrain[t][0]:.4f}, {macrostrain[t][1]:.4f}, {macrostrain[t][2]:.4f}]"
        titles.append([f'Prediction{extra_title}', macro_title, 'Error'])

    # Plotting homogenized curves
    plotHomogenizedStress(mesh_ipArea_single, stresses, macrostrain, homstress, homStressNorm, dirstring, m, settings, extra_title, homstress_pred)

    if settings['plot_fullfield']:  # This is set hardcoded in load_model
        print("Plotting full field predictions..")
        # Plotting full field
        # initialize mergers
        merger_epsx = PdfMerger()  # initialize merger
        merger_epsy = PdfMerger()
        merger_epsxy = PdfMerger()
        merger_epspeq = PdfMerger()
        merger_sigx = PdfMerger()  # initialize merger
        merger_sigy = PdfMerger()
        merger_sigxy = PdfMerger()
        mergers = [merger_epsx, merger_epsy, merger_epsxy, merger_epspeq, merger_sigx, merger_sigy, merger_sigxy]

        # Find epsilon bounds & format data
        preds = np.empty((timesteps, strains[0].shape[0], strains[0].shape[1]))
        targets = np.empty((timesteps, strains[0].shape[0], strains[0].shape[1]))
        epspeq_preds = np.empty((timesteps, epspeqs[0].shape[0]))
        epspeq_targets = np.empty((timesteps, epspeqs[0].shape[0]))

        for t in range(timesteps):
            preds[t] = strains[t].detach().cpu().numpy()
            epspeq_preds[t] = epspeqs[t].detach().cpu().numpy()

            targets[t] = targets_all_steps[t]

            if settings['stress_as_input_field'] is not True and settings['epspeq_feature']:
                try:
                    epspeq_targets[t] = epspeq_all_steps[t]
                except: # Dirty fix, when using gpu, it is called with both the cpu and gpu.
                    epspeq_targets[t] = epspeq_all_steps[t].cpu()

        all_bound_values = []
        errors = preds - targets
        for feat in range(3):
            max_main = max(np.amax(preds[:, :, feat]), np.amax(targets[:, :, feat]))
            min_main = min(np.amin(preds[:, :, feat]), np.amin(targets[:, :, feat]))


            max_err = np.amax(errors[:, :, feat])
            min_err = np.amin(errors[:, :, feat])
            err_bnd = max(abs(max_err), abs(min_err))

            bound_main = [min_main, max_main]
            bound_err = [-err_bnd, err_bnd]
            bound_values = [bound_main, bound_main, bound_err]
            all_bound_values.append(bound_values)

        # Find sigma bounds & format data
        sig_target = np.empty((timesteps, strains[0].shape[0], strains[0].shape[1]))
        stress_preds = np.empty((timesteps, strains[0].shape[0], strains[0].shape[1]))

        for t in range(timesteps):
            sig_target[t] = elemStressNorm.denormalize(norm_stresses[m][(3 * num_nodes) * t: (3 * num_nodes) * (t + 1)].reshape((strains[0].shape[0], strains[0].shape[1])))
            stress_preds[t] = stresses[t]

        sig_all_bound_values = []
        sig_errors = stress_preds - sig_target
        for feat in range(3):
            max_main = max(np.amax(stress_preds[:, :, feat]), np.amax(sig_target[:, :, feat]))
            min_main = min(np.amin(stress_preds[:, :, feat]), np.amin(sig_target[:, :, feat]))

            max_err = np.amax(sig_errors[:, :, feat])
            min_err = np.amin(sig_errors[:, :, feat])
            err_bnd = max(abs(max_err), abs(min_err))

            bound_main = [min_main, max_main]
            bound_err = [-err_bnd, err_bnd]
            bound_values = [bound_main, bound_main, bound_err]
            sig_all_bound_values.append(bound_values)

        # Find epspeqs bounds & format data
        if settings['stress_as_input_field'] is not True and settings['epspeq_feature']:
            max_epspeq = max(np.amax(epspeq_preds), np.amax(epspeq_targets))
            min_epspeq = min(np.amin(epspeq_preds), np.amin(epspeq_targets))

            epspeq_errors = epspeq_preds - epspeq_targets
            epspeq_max_err = np.amax(epspeq_errors)
            epspeq_min_err = np.amin(epspeq_errors)
            epspeq_err_bnd = max(abs(epspeq_max_err), abs(epspeq_min_err))

            epspeq_bound_main = [min_epspeq, max_epspeq]
            epspeq_bound_err = [-epspeq_err_bnd, epspeq_err_bnd]
            epspeq_bound_values = [epspeq_bound_main, epspeq_bound_main, epspeq_bound_err]

        for t in range(timesteps + 1):  #+1 to start with 0
            # for t in range(2):
            if final_only and t != timesteps:  # Option to only print last step, saves lot of time.
            # if final_only and t not in [8, 12, 22, 25]:  # Option to only print last step, saves lot of time.
            #     print(f"Step {t} to pdf (skipped)")
                continue
            print(f"Step {t} to pdf")

            # color_style_plots = ['jet', 'jet', 'bwr']
            color_style_plots = ['Spectral', 'Spectral', 'bwr']

            if t == 0:  # Plot 0 values
                tmp_macro_title = r"$\varepsilon^{\Omega} =$" + f"[{0.0:.4f}, {0.0:.4f}, {0.0:.4f}]"
                for feat in range(3):
                    filename = f"{dirstring}/{m}_-1_{feat}.pdf"

                    values = [np.zeros_like(preds[t, :, feat]), np.zeros_like(targets[t, :, feat]), np.zeros_like(errors[t, :, feat])]
                    labels = [fr'$\varepsilon_{int(feat)}$', fr'$\varepsilon_{int(feat)}$', fr'$\varepsilon_{int(feat)}$']

                    color_plot_combined(tri_mesh, values, [f'Prediction{extra_title}', tmp_macro_title, 'Error'], filename, all_bound_values[feat],
                                        color_style_plots, labels)

                    mergers[feat].append(filename)
                    os.remove(filename)

                # Epspeq
                if settings['stress_as_input_field'] is not True and settings['epspeq_feature']:  #(stress to stress no material model)
                    filename = f"{dirstring}/{m}_-1_epspeq.pdf"
                    values = [np.zeros_like(epspeq_preds[t, :]), np.zeros_like(epspeq_targets[t, :]), np.zeros_like(epspeq_errors[t, :])]
                    labels = [fr'$\varepsilon^p_{{eq}}$', fr'$\varepsilon^p_{{eq}}$', fr'$\varepsilon^p_{{eq}}$']
                    color_plot_combined(tri_mesh, values, [f'Prediction{extra_title}', tmp_macro_title, 'Error'], filename, epspeq_bound_values, color_style_plots, labels)

                    mergers[3].append(filename)
                    os.remove(filename)

                # Stresses
                for feat in range(3):
                    filename = f"{dirstring}/{m}_-1_{feat}_sig.pdf"

                    values = [np.zeros_like(stress_preds[t, :, feat]), np.zeros_like(sig_target[t, :, feat]), np.zeros_like(sig_errors[t, :, feat])]
                    labels = [fr'$\sigma_{int(feat)}$', fr'$\sigma_{int(feat)}$', fr'$\sigma_{int(feat)}$']

                    color_plot_combined(tri_mesh, values, [f'Prediction{extra_title}', tmp_macro_title, 'Error'], filename, sig_all_bound_values[feat], color_style_plots, labels)

                    mergers[4 + feat].append(filename)
                    os.remove(filename)
            else:
                cur_t = t - 1
                # Strains
                for feat in range(3):
                    filename = f"{dirstring}/{m}_{cur_t}_{feat}.pdf"

                    values = [preds[cur_t, :, feat], targets[cur_t, :, feat], errors[cur_t, :, feat]]
                    labels = [fr'$\varepsilon_{int(feat)}$', fr'$\varepsilon_{int(feat)}$', fr'$\varepsilon_{int(feat)}$']

                    color_plot_combined(tri_mesh, values, titles[cur_t], filename, all_bound_values[feat], color_style_plots, labels)

                    mergers[feat].append(filename)
                    os.remove(filename)

                # Epspeq
                if settings['stress_as_input_field'] is not True and settings['epspeq_feature']:  #(stress to stress no material model)
                    filename = f"{dirstring}/{m}_{cur_t}_epspeq.pdf"
                    values = [epspeq_preds[cur_t, :], epspeq_targets[cur_t, :], epspeq_errors[cur_t, :]]
                    labels = [fr'$\varepsilon^p_{{eq}}$', fr'$\varepsilon^p_{{eq}}$', fr'$\varepsilon^p_{{eq}}$']
                    color_plot_combined(tri_mesh, values, titles[cur_t], filename, epspeq_bound_values, color_style_plots, labels)

                    mergers[3].append(filename)
                    os.remove(filename)

                # Stresses
                for feat in range(3):
                    filename = f"{dirstring}/{m}_{cur_t}_{feat}_sig.pdf"

                    values = [stress_preds[cur_t, :, feat], sig_target[cur_t, :, feat], sig_errors[cur_t, :, feat]]
                    labels = [fr'$\sigma_{int(feat)}$', fr'$\sigma_{int(feat)}$', fr'$\sigma_{int(feat)}$']

                    color_plot_combined(tri_mesh, values, titles[cur_t], filename, sig_all_bound_values[feat], color_style_plots, labels)

                    mergers[4 + feat].append(filename)
                    os.remove(filename)

            # Close all plots (otherwise all will continuously consume memory)
            plt.close('all')

        mergers[0].write(f"{dirstring}/{m}_epsx{extra_title}.pdf")
        merger_epsx.close()
        mergers[1].write(f"{dirstring}/{m}_epsy{extra_title}.pdf")
        merger_epsy.close()
        mergers[2].write(f"{dirstring}/{m}_epsxy{extra_title}.pdf")
        merger_epsxy.close()
        mergers[3].write(f"{dirstring}/{m}_epspeq{extra_title}.pdf")
        merger_epspeq.close()
        mergers[4].write(f"{dirstring}/{m}_sigx{extra_title}.pdf")
        merger_sigx.close()
        mergers[5].write(f"{dirstring}/{m}_sigy{extra_title}.pdf")
        merger_sigy.close()
        mergers[6].write(f"{dirstring}/{m}_sigxy{extra_title}.pdf")
        merger_sigxy.close()



# Works, but outdated. Unused variables.
def plotHomogenizedStress(ipArea, stress_list, strain_list, homstress, homStressNorm, dirstring, m, settings,
                              extra_title='', homstress_pred=None):

    stresses = homstress_pred.cpu()   # move to cpu when using GPU to avoid plotting error

    plt.figure()
    components = ['x', 'y', 'xy']
    unnorm_homstress = homStressNorm.denormalize(homstress)

    for feat in range(3):
        # True
        if feat == 0:
            newcurve = plt.plot(np.append(0,strain_list[:,feat]), np.append(0, unnorm_homstress[:,feat]), label='True')
        else:
            newcurve = plt.plot(np.append(0,strain_list[:,feat]), np.append(0, unnorm_homstress[:,feat]))
        # Prediction
        newcurve = plt.plot(np.append(0, strain_list[:, feat]), np.append(0, stresses[:, feat]), '--', label=components[feat], c=newcurve[0].get_color())

    plt.xlabel(r"$\varepsilon_i$")
    plt.ylabel(r"$\sigma_i$")

    plt.legend(fontsize=6) #, ncol=3)
    plt.savefig(f"{dirstring}/{m}_homogenized{extra_title}.pdf", bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')

    ### TEMPORARILY SAVING TO FOLDER
    with open(f"{dirstring}/sig_eps_{m}", 'w') as f:
        for strain0, stresstrue0, stresspred0, strain1, stresstrue1, stresspred1, strain2, stresstrue2, stresspred2 in zip(strain_list[:,0], unnorm_homstress[:,0], stresses[:,0], strain_list[:,1], unnorm_homstress[:,1], stresses[:,1], strain_list[:,2], unnorm_homstress[:,2], stresses[:,2]):
            f.write(f"{strain0} {stresstrue0} {stresspred0} {strain1} {stresstrue1} {stresspred1} {strain2} {stresstrue2} {stresspred2}\n")

    return

