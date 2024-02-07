"""
This script is used to obtain results based on pre-trained GNN models.

An experiment is selected below, and the specific experiment settings are read from the config file in the experiments/<name> folder.
Change the name of the exp_config to be run.
For example, we've changed "../10_testquantity/exp_config_69" to "../10_testquantity/exp_config", which is now used.

For some experiments, additional hardcoded datasets & parameters are set later in the file.
Outdated experiments have been removed.
Some tests do not cancel when finished, and continue to the plotting phase, where errors can occur depending on e.g. batch settings.
"""

import os
import time

import numpy as np
import torch
import torch_geometric
from matplotlib import pyplot as plt
from matplotlib import rc
from torch_geometric.loader import DataLoader

from GNN_MPL import GNN
from graphDatasets import createDataset
from load_data import load_data
from lossFunc import computeLoss, computeNorm_t
from plotUtils import rve_unroll_visual
from readConfig import read_config, get_batch_settings, printConfig
from readExpConfig import read_exp_config, print_exp_config
import normUtils

rc('text', usetex=True)
plt.style.use(['science', 'bright'])


init_tensor_dtype = torch.float32
torch.set_default_dtype(init_tensor_dtype)

use_cuda = torch.cuda.is_available()

if use_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

"""
Manually select experiment
"""
# experiment = '2_unloading'
# experiment = '3_uni_stiff_conv'
# experiment = '4_time_extrap'
experiment = '5_plot'
# experiment = '10_testquantity'
# experiment = '13_comptime'

"""
Load settings
"""

configuration = read_exp_config(f"../experiments/{experiment}/exp_config")

model_dirpath = f"../results/hyperparam_studies/{configuration['model']}"

test_type = configuration['type']

settings = read_config(f'{model_dirpath}/ConfigGNN')
random_seed = settings['seed']
settings = get_batch_settings(settings, 0)
settings['seed'] = random_seed
# settings = read_config(f'{model_dirpath}/ConfigGNN_transfer')

settings['plot_fullfield'] = configuration['plot_field']

dirstring = f"../experiments/{experiment}/{configuration['name']}"

compare_data = False
representative_samples = True
if configuration['representative_based_on'] == 'homstress':
    repr_type = 'hom'
elif configuration['representative_based_on'] == 'field':
    repr_type = 'full_field'

if configuration['trim']:
    settings['trainsize'] = 10
    settings['trainend'] = 10
    settings['validationend'] = 20
    settings['transferend'] = 21

if test_type == 'unloading':
    # Proper unloading
    print(f"Make sure to plot using representative samples!")
    settings['transfersteps'] = 30
    settings['transferdir'] = f'testset/fib1-9_mesh2000_t30_4_unload'
    settings['transferend'] = 2020
elif test_type == 'stiff_meshsize':  # V, no duplicate steps.
    settings['transfersteps'] = 1
    settings['transferdir'] = f'stiff/fib{configuration["extrapolate_num"]}_mesh500_t1_uni'
    settings['transferend'] = 520
    # If not using GP 1-9:  (so this can be used for both " " and "_gp" when having e.g. 1-3 voids. Doesn't affect test set.)
    settings['meshdir'] = f'../meshes/fib1-9_mesh6000_t25_gp'
    settings['datafolder'] = f'../data/fib1-9_mesh6000_t25_gp'

    dirstring += f"/{configuration['extrapolate_num']}"
elif test_type == 'time_extrap':
    settings['transfersteps'] = 50
    settings['transferend'] = 2020
    settings['transferdir'] = f'testset/fib1-9_mesh2000_t50_gp'
    dirstring += "_norm"
    if configuration['time_exrap_baseline']:
        dirstring += "_baseline"
elif test_type == 'plot':  # X
    # # Default!!
    # settings['transfersteps'] = 25
    # settings['transferdir'] = f'testset/fib1-9_mesh2000_t25_gp'
    # settings['transferend'] = 2020

    # # Plot large mesh instead
    settings['transfersteps'] = 25
    settings['transferdir'] = f'testset/fib49_mesh500_t25_gp'
    settings['transferend'] = 520

elif test_type == 'testquantity':  # X
    settings['transfersteps'] = 25
    settings['transferdir'] = f'testset/fib1-9_mesh2000_t25_gp'
    settings['transferend'] = 2020

settings['transferdata'] = f"../data/{settings['transferdir']}"
settings['transfermesh'] = f"../meshes/{settings['transferdir']}"
empty_micro = False

if test_type == 'comptime':
    settings['transfersteps'] = 25
    batch_size_comptime = 1

    settings['transferdir'] = f"../dogbone/m{configuration['extrapolate_num']}_t{settings['transfersteps']}"
    settings['transferdata'] = f"{settings['transferdir']}/data"
    settings['transfermesh'] = f"{settings['transferdir']}/meshes"

    # Get the number of elements in the micro by looking at the first number in the line after "$Elements"
    track_now = False
    with open(f"{settings['transfermesh']}/m_0.msh") as meshdata:
        for k, line in enumerate(meshdata):
            array = line.split()
            if track_now:
                empty_micro = int(array[0])
            if array[0] == "$Elements":
                track_now = True
            else:
                track_now = False

    # Get number of macro elements (strain paths)
    with open(f"{settings['transferdata']}/macrostrain.data", 'r') as data:
        num_macro_elems = len(data.readlines())

    settings['transferend'] = num_macro_elems + 20  # 20 = because of trimmed train + validation samples loaded.

settings['batch_size'] = 1
np.random.seed(settings['seed'])

# Get normalization parameters from original model
norm_scalers = normUtils.readNormFile(f'{model_dirpath}/normbounds')
geomFeatNorm, edgeFeatNorm, _, _, _, _, _, _ = norm_scalers
normclassexists = True

# Load data and normalize
fibers, macro_vec_train, macro_vec_transfer, eps_set, eps_set_noise, epspeq_set_noise, macroStrainNorm, elemStrainNorm, elemepspNorm, elemepspeqNorm, homstress_train, homstress_transfer, homStressNorm, normalized_stresses, elemStressNorm = load_data(settings, norm_scalers, configuration['trim'], empty_micro)


# To use unrolled with the real material model, steps_ahead = 1 and manual stepping applied.
steps_ahead = settings['steps_ahead']

# Compute number of samples per mesh
sets_per_graph = 1

faulty_meshes = []
"""
For every #num fibers considered ..
"""
# for trainfibnum in settings['fibers_considered']:
trainfibnum = settings['fibers_considered'][0]

print(f"Fibers: {trainfibnum}, with meshes from meshdir: {settings['meshdir']} ")

strain_components = 3  # (eps_x, eps_y, eps_xy)
pred_components = 3  # (eps_x, eps_y, eps_xy)

geom_features = trainfibnum * 2
base_inputfeats = geom_features
num_features = geom_features + strain_components
if settings['load_vec_as_feature']:
    num_features += 3
    base_inputfeats += 3
if settings['epspeq_feature']:
    num_features += 1
if settings['numvoid_feature']:
    num_features += 1

pred_features = strain_components       # 3
if settings['fullfield_stress']:        # include full field stress as additional output
    pred_features += pred_components    # 3 + 3

"""
Create graph & set features and targets
"""
datasets, mesh_elem_nodes_set, mesh_node_coords_set, mesh_ipArea_set, geomFeatNorm, edgeFeatNorm = createDataset(settings, init_tensor_dtype, macro_vec_train, homstress_train, macro_vec_transfer, homstress_transfer, fibers, trainfibnum, strain_components, pred_components, elemStrainNorm, elemepspeqNorm, eps_set, epspeq_set_noise, geom_features, normalized_stresses, normclassexists, geomFeatNorm, edgeFeatNorm)

"""
Set-up dataloader
"""
del(eps_set)
del(eps_set_noise)
del(epspeq_set_noise)

print("Datasets created. Creating dataloaders..")
torch_geometric.seed_everything(settings['seed'])
trainstart_cur = settings['trainend'] - settings['trainsize']


val_dataLoad = DataLoader(
    datasets[settings['trainend'] * sets_per_graph:settings['validationend'] * sets_per_graph],
    batch_size=settings['batch_size'], shuffle=False)

# For plotting: Repeat above, but for training size = 1 and no shuffle in training
plot_train_dataLoad = DataLoader(datasets[trainstart_cur * sets_per_graph:settings['trainend'] * sets_per_graph],
                                 batch_size=1, shuffle=False)
plot_val_dataLoad = DataLoader(
    datasets[settings['trainend'] * sets_per_graph:settings['validationend'] * sets_per_graph], batch_size=1,
    shuffle=False)

if settings['transferend'] != settings['validationend']:
    plot_transf_dataLoad = DataLoader(
        datasets[settings['validationend'] * sets_per_graph:settings['transferend'] * sets_per_graph], batch_size=1,
        shuffle=False)
if test_type == 'testquantity':
    plot_transf_dataLoad = DataLoader(
        datasets[settings['validationend'] * sets_per_graph:settings['transferend'] * sets_per_graph], batch_size=32,
        shuffle=False)
if test_type == 'comptime':
    plot_transf_dataLoad = DataLoader(
        datasets[settings['validationend'] * sets_per_graph:settings['transferend'] * sets_per_graph], batch_size=batch_size_comptime,
        shuffle=False)

"""
Load network
"""
layers = settings['layers_arr'][0]
print(f"L:{layers} HN:{settings['hidden_nodes']} act:{settings['activation_func']}")

"""
Initializing class
"""
settings['w_init'] = 'kaiming_uniform'  # Doesn't matter, but required for init
best_model = GNN(num_features, pred_features, layers, settings, device, elemStrainNorm, elemepspeqNorm,homStressNorm,elemStressNorm).to(device)
best_model.train(mode=False)
best_model.settings['run_material'] = True  # If the material model not used in training, do still use it for plotting

if not test_type == 'testquantity':
    if use_cuda:
        best_model.load_state_dict(torch.load(model_dirpath + "/model"))
    else:
        best_model.load_state_dict(torch.load(model_dirpath + "/model", map_location=torch.device('cpu')))

    """
    Make directory
    """
    if not test_type == 'meshes' and not test_type == 'stiff_meshsize':
        try:
            print(f"Creating directory {dirstring}")
            os.mkdir(dirstring)
        except:
            print("#############################################")
            print("Unroll directory could not be made.")
            print("This might cause errors or overwrite something!\n Continuing anyway")

"""
Find representative samples - those with the median loss of the full dataset
"""

def findRepresentativeSamples(mode='plot'):
    loads = [plot_train_dataLoad, plot_val_dataLoad, plot_transf_dataLoad]
    labels = ['train', 'val', 'test']
    repr_samples = []

    # Compute the loss for all samples, print which samples have the highest and the lowest error, to plot them after.
    best_model.eval()
    with torch.no_grad():
        for i, dataset in enumerate(loads):
            if configuration['plot_representative'][i] == 0:
                repr_samples.append([])
                continue
            else:       # Loop over dataset and get representative samples
                losses = np.zeros(len(dataset))
                strain_field_losses = np.zeros(len(dataset))
                stress_field_losses = np.zeros(len(dataset))
                hom_losses = np.zeros(len(dataset))
                strain_field_losses_norm = np.zeros(len(dataset))
                stress_field_losses_norm = np.zeros(len(dataset))
                hom_losses_norm = np.zeros(len(dataset))
                losses_val = 0.0
                strain_field_losses_val = 0.0
                stress_field_losses_val = 0.0
                hom_losses_val = 0.0
                strain_field_losses_val_norm = 0.0
                stress_field_losses_val_norm = 0.0
                hom_losses_val_norm = 0.0
                for j, data in enumerate(dataset):
                    if j % 16 == 0:     # counts in batches, so text output needs to be multiplied with batch size.
                        print(f"{labels[i]}: Predicting batch: {j}")

                    loss, strain_field_loss, stress_field_loss, hom_loss, strain_field_loss_norm, stress_field_loss_norm, hom_loss_norm = computeLoss(best_model, data, settings, len(dataset), device, elemStrainNorm, elemStressNorm, homStressNorm)
                    # loss, field_loss, hom_loss = computeLoss(best_model, data, settings, len(dataset), device, elemStrainNorm, elemStressNorm, homStressNorm)
                    losses[j] = float(loss)
                    strain_field_losses[j] = float(strain_field_loss)
                    stress_field_losses[j] = float(stress_field_loss)
                    hom_losses[j] = float(hom_loss)
                    strain_field_losses_norm[j] = float(strain_field_loss_norm)
                    stress_field_losses_norm[j] = float(stress_field_loss_norm)
                    hom_losses_norm[j] = float(hom_loss_norm)
                    # Sum, divide by dataset length to make average
                    losses_val += float(loss) / len(dataset)
                    strain_field_losses_val += float(strain_field_loss) / len(dataset)
                    stress_field_losses_val += float(stress_field_loss) / len(dataset)
                    hom_losses_val += float(hom_loss) / len(dataset)
                    strain_field_losses_val_norm += float(strain_field_loss_norm) / len(dataset)
                    stress_field_losses_val_norm += float(stress_field_loss_norm) / len(dataset)
                    hom_losses_val_norm += float(hom_loss_norm) / len(dataset)

                if representative_samples:
                    # Sort by difference from median, get arguments
                    if repr_type == 'full_field':
                        # Full field
                        diff_array = np.abs(losses - np.median(losses)).argsort()
                    elif repr_type == 'hom':
                        # Homogenized
                        diff_array = np.abs(hom_losses - np.median(hom_losses)).argsort()
                else:   # Just take the minimum ones.
                    if repr_type == 'full_field':
                        diff_array = losses.argsort()
                    elif repr_type == 'hom':
                        diff_array = hom_losses.argsort()

                    diff_array = losses.argsort()
                repr_samples.append(diff_array[0:configuration['plot_representative'][i]])

                print(f"Loss: {losses_val}, strain field loss: {strain_field_losses_val},  stress field loss: {stress_field_losses_val}, hom_loss: {hom_losses_val}")
                with open(f"{dirstring}/{labels[i]}_loss.txt", 'w') as f:
                    f.write(f"{losses_val},{strain_field_losses_val},{stress_field_losses_val},{hom_losses_val},{strain_field_losses_val_norm},{stress_field_losses_val_norm},{hom_losses_val_norm}\n")
                    f.write(f"{np.sqrt(losses_val)},{np.sqrt(strain_field_losses_val)},{np.sqrt(stress_field_losses_val)},{np.sqrt(hom_losses_val)},{np.sqrt(strain_field_losses_val_norm)},{np.sqrt(stress_field_losses_val_norm)},{np.sqrt(hom_losses_val_norm)}\n")
                if mode == 'plot':
                    with open(f"{dirstring}/all_{labels[i]}_losses.txt", 'w') as f:
                        for loss, strain_field_loss, stress_field_loss, hom_loss, strain_field_loss_norm, stress_field_loss_norm, hom_loss_norm in zip(losses, strain_field_losses, stress_field_losses, hom_losses, strain_field_losses_norm, stress_field_losses_norm, hom_losses_norm):
                            f.write(f"{loss}, {strain_field_loss}, {stress_field_loss}, {hom_loss}, {strain_field_loss_norm}, {stress_field_loss_norm}, {hom_loss_norm}\n")

    configuration['plot_train'] = repr_samples[0]
    configuration['plot_val'] = repr_samples[1]
    configuration['plot_extra'] = repr_samples[2]
    print(f"Repr.configs: {configuration['plot_train']}, {configuration['plot_val']}, {configuration['plot_extra']}")

# Find the samples which are representative in each dataset, and replace the plot config with those
if np.sum(configuration['plot_representative']) != 0:
    findRepresentativeSamples(mode='plot')

"""
Testquantity model: General one used for different results. Stores errors and plots.
"""

if test_type == 'testquantity':
    # Set plot_representative to nonzero so the test set is computed
    configuration['plot_representative'] = [0,0,1]

    if settings['transferdir'][-3:] == 'gp':
        extra_text = 'gp'
    else:
        extra_text = '4'

    # Loop through models
    print(f"Printing the following models:")
    print(os.listdir(f"../results/hyperparam_studies/{configuration['name']}"))
    for model_dir in os.listdir(f"../results/hyperparam_studies/{configuration['name']}"):
        if model_dir[:5] == 'slurm' or model_dir[:4] == 'fail' or model_dir[:3] == 'tmp' or model_dir[-4:] == '.png' or model_dir[-4:] == '.pdf':
            continue
        print(f"Model: {model_dir}")
        # Initialize settings
        settings = read_config(f"../results/hyperparam_studies/{configuration['name']}/{model_dir}/ConfigGNN")
        random_seed = settings['seed']
        settings = get_batch_settings(settings, 0)

        settings['seed'] = random_seed
        settings['w_init'] = 'kaiming_uniform'  # Doesnt matter, but required for initializing the GNN

        # Compute the losses and plot 2 representative samples

        ### Make directory and setup model. Choose how to name the resulting folder
        # Learncurve
        # dirstring = f"../experiments/{experiment}/{configuration['name']}/size_{settings['trainsize']}_{random_seed}_{extra_text}"
        # Xi based:
        # dirstring = f"../experiments/{experiment}/{configuration['name']}/xi_{settings['xi_strainfield']:.2f}_{settings['xi_stressfield']:.2f}_{settings['xi_stresshom']:.2f}"
        # MPL based
        # dirstring = f"../experiments/{experiment}/{configuration['name']}/MPL_{settings['layers_arr'][0]}_{random_seed}"
        # Dropout based
        # dirstring = f"../experiments/{experiment}/{configuration['name']}/drop_{settings['dropout']}"
        # Others, to be implemented
        # Combined Xi, MPL and dropout:
        dirstring = f"../experiments/{experiment}/{configuration['name']}/combined_{settings['layers_arr'][0]}MPL_{settings['dropout']}drop_{settings['xi_strainfield']:.2f}_{settings['xi_stressfield']:.2f}_{settings['xi_stresshom']:.2f}_{extra_text}"

        # Different # layers / dropout etc, so init new model.
        best_model = GNN(num_features, pred_features, settings['layers_arr'][0], settings, device, elemStrainNorm, elemepspeqNorm,homStressNorm,elemStressNorm).to(device)
        best_model.train(mode=False)
        best_model.settings['run_material'] = True  # If material model not used in training, do still use it for plotting
        # Load model state
        best_model.load_state_dict(torch.load(f"../results/hyperparam_studies/{configuration['name']}/{model_dir}/model", map_location=device))

        # Make directory
        for i in range(9):
            try:
                print(f"Creating directory {dirstring}")
                os.mkdir(dirstring)
                break
            except:
                print("New folder could not be created.")
                dirstring = dirstring[:-2] + f"_{i+1}"

        # Compute the losses
        findRepresentativeSamples(mode='loss')

        # Copy the config file
        printConfig(f'{dirstring}/ConfigGNN', settings)
    exit()

"""
Compare data: Compute unnormalized loss on transfer dataset
"""

if compare_data:
    losses_ar = np.zeros(len(plot_transf_dataLoad))
    hom_losses_ar = np.zeros(len(plot_transf_dataLoad))
    losses = 0.0
    hom_losses = 0.0
    ## Compute the loss for all samples, print which samples have the highest and the lowest error, to plot them after.
    best_model.eval()
    with torch.no_grad():
        for i, data in enumerate(plot_transf_dataLoad):
            if i % 50 == 0:
                print(f"Predicting: {i}")
            loss, field_loss, hom_loss = computeLoss(best_model, data, settings, len(plot_transf_dataLoad), device, elemStrainNorm, elemStressNorm, homStressNorm)
            losses_ar[i] = float(field_loss)
            hom_losses_ar[i] = float(hom_loss)
            losses += float(loss)
            hom_losses += float(hom_loss)
    print(f"Loss: {losses}, hom_loss: {hom_losses}")
    with open(f"{dirstring}/transf_losses.txt", 'w') as f:
        f.write(f"{losses},{hom_losses}\n")
        f.write(f"{np.sqrt(losses)},{np.sqrt(hom_losses)}\n")
    with open(f"{dirstring}/all_transf_losses.txt", 'w') as f:
        for loss, hom_loss in zip(losses_ar, hom_losses_ar):
            f.write(f"{loss},{hom_loss}\n")


"""
Compute stiffness
"""

def randVector():
    """
    Computing random load vector: A 3D unit vector with random 'direction'
    Based on answer https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d
    (𝑥,𝑦,𝑧)=(sqrt(1−𝑧^2)cos𝜃,sqrt(1−𝑧^2)sin𝜃,𝑧)
    """
    theta = torch.rand(1) * 2 * torch.pi
    z0 = torch.rand(1) * 2 - 1
    x = torch.sqrt(1-z0**2)*torch.cos(theta)
    y = torch.sqrt(1-z0**2)*torch.sin(theta)
    return torch.tensor([x.item(), y.item(), z0.item()])


def print_Stiffness():
    # Obtain stiffnesses for all samples in transfer dataset
    stiffnesses = np.zeros(len(plot_transf_dataLoad))  # Only first component
    for i, data in enumerate(plot_transf_dataLoad):
        if i % 50 == 0:
            print(f"Computing stiffness: {i}")
        data = data.clone().to(device)

        # Compute stiffness
        if configuration['stiff_mode'] == 'randdir':
            strainin = randVector() / 10000  # Make strain very small to minimize plasticity. (Always some plasticities after GNN, even for all 0)
        else:  # assumes stiff_mode == 'zero':
            strainin = torch.tensor([0., 0., 0.])

        stiffness = best_model.getStiffness(data, macroStrainNorm, strainin)

        stiffnesses[i] = stiffness[0, 0]

    # Print stiffnesses
    print(f"Printing stiffnesses")
    with open(f"{dirstring}/stiffnesses_{configuration['extrapolate_num']}.txt", 'w') as f:
        for stiff in stiffnesses:
            f.write(f"{stiff}\n")

if test_type == 'stiff_meshsize':       # Print stiffness for many datasets.
    # Loop over mesh sizes and compute stiffnesses
    transfer_nums = [1, 4, 9, 16,25,36,49,64]
    for transf_num in transfer_nums:
        dirstring = f"../experiments/{experiment}/{configuration['name']}/{transf_num}"
        print(f"Creating directory {dirstring}")
        os.mkdir(dirstring)

        settings['transferdir'] = f'stiff/fib{transf_num}_mesh500_t1_uni'
        settings['transferdata'] = f"../data/{settings['transferdir']}"
        settings['transfermesh'] = f"../meshes/{settings['transferdir']}"

        # Load data and normalize
        fibers, macro_vec_train, macro_vec_transfer, eps_set, eps_set_noise, epspeq_set_noise, macroStrainNorm, elemStrainNorm, elemepspNorm, elemepspeqNorm, homstress_train, homstress_transfer, homStressNorm, normalized_stresses, elemStressNorm = load_data(
            settings, norm_scalers, configuration['trim'])
        # Create graphs
        datasets, mesh_elem_nodes_set, mesh_node_coords_set, mesh_ipArea_set, geomFeatNorm, edgeFeatNorm = createDataset(
            settings, init_tensor_dtype, macro_vec_train, homstress_train, macro_vec_transfer, homstress_transfer,
            fibers,
            trainfibnum, strain_components, pred_components, elemStrainNorm, elemepspeqNorm, eps_set, epspeq_set_noise, geom_features,
            normalized_stresses, normclassexists, geomFeatNorm, edgeFeatNorm)

        plot_transf_dataLoad = DataLoader(
            datasets[settings['validationend'] * sets_per_graph:settings['transferend'] * sets_per_graph], batch_size=1,
            shuffle=False)

        print_Stiffness()
        print_exp_config(f"{dirstring}/exp_config", configuration)


elif configuration['print_stiffness']:      # Print stiffness for single dataset
    print_Stiffness()


"""
Compute error
"""
def plot_t_error(errors, settings, type, label):  # Type = Average or Max
    # Plotting mean
    mean_time_error = np.mean(errors, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    x_arr = np.arange(len(mean_time_error)) + 1
    x_arr_expanded = np.broadcast_to(x_arr, (errors.shape))

    colors = ['#1b9e77', '#d95f02']

    if settings['timesteps'] == settings['transfersteps']:
        # ax.fill_between(x_arr, mean_time_error - std_time_error, mean_time_error + std_time_error, alpha=0.3,label='Standard deviation', facecolor='gray')
        ax.scatter(x_arr_expanded, errors, s=4, color=colors[0], alpha=0.15)  # FEM
        ax.plot(x_arr, mean_time_error, label='Mean error')
    else:
        # Training zone
        # ax.fill_between(x_arr[:settings['timesteps']],
        #                 mean_time_error[:settings['timesteps']] - std_time_error[:settings['timesteps']],
        #                 mean_time_error[:settings['timesteps']] + std_time_error[:settings['timesteps']], alpha=0.3,
        #                 label='Standard deviation', facecolor=colors[0])
        ax.scatter(x_arr_expanded[:,:settings['timesteps']], errors[:,:settings['timesteps']], s=4, color=colors[0], alpha=0.15)  # FEM
        ax.plot(x_arr[:settings['timesteps']], mean_time_error[:settings['timesteps']], label='Mean error',
                color=colors[0])

        # Extrapolation zone
        # ax.fill_between(x_arr[settings['timesteps'] - 1:],
        #                 mean_time_error[settings['timesteps'] - 1:] - std_time_error[settings['timesteps'] - 1:],
        #                 mean_time_error[settings['timesteps'] - 1:] + std_time_error[settings['timesteps'] - 1:],
        #                 alpha=0.3, facecolor=colors[1])
        ax.scatter(x_arr_expanded[:,settings['timesteps']:], errors[:,settings['timesteps']:], s=4, color=colors[1], alpha=0.15)  # FEM
        ax.plot(x_arr[settings['timesteps'] - 1:], mean_time_error[settings['timesteps'] - 1:], label='Extrapolation',
                color=colors[1])

    plt.xlabel('Timestep')
    # plt.ylabel(f'{type} ' + r'$ ||\hat{\varepsilon}^{\omega}-\varepsilon^{\omega}||^2 $')   # MSE
    plt.ylabel(label)    # MAE
    plt.legend(loc='upper left')
    plt.savefig(f'{dirstring}/t_{type}_error.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
    plt.savefig(f'{dirstring}/t_{type}_error.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')


if test_type == 'time_extrap':
    with torch.no_grad():
        """
        Computing errors per timestep.
        Note that this is done for the full field unnormalized strain!
        """
        print(f"Computing errors")
        num_samples = len(plot_transf_dataLoad)

        mean_eps_field = np.zeros((num_samples, settings['transfersteps']))
        max_eps_field = np.zeros((num_samples, settings['transfersteps']))
        mean_sig_field = np.zeros((num_samples, settings['transfersteps']))
        max_sig_field = np.zeros((num_samples, settings['transfersteps']))
        mean_homsig = np.zeros((num_samples, settings['transfersteps']))
        max_homsig = np.zeros((num_samples, settings['transfersteps']))

        for i, data in enumerate(plot_transf_dataLoad):
            if i % 100 == 0:
                print(f"Predicting sample: {i} / {num_samples}")
            data = data.clone().to(device)

            # Compute norm error. Pass macroStrainNorm = None to have the baseline be 0 strain
            # macroStrainNorm = None
            mean_eps_field[i, :], max_eps_field[i, :], mean_sig_field[i, :], max_sig_field[i, :], mean_homsig[i, :], max_homsig[i, :] = computeNorm_t(best_model, data, device, elemStrainNorm, elemStressNorm, homStressNorm, comp_baseline = configuration['time_exrap_baseline'], macroStrainNorm=macroStrainNorm)

        print(f"Printing timestep errors")
        with open(f"{dirstring}/eps_t_errors.txt", 'w') as f:
            for i in range(num_samples):
                string = f"{mean_eps_field[i, 0]}"
                for j in range(settings['transfersteps']-1):
                    string += f" {mean_eps_field[i, j+1]}"
                string += f"\n"
                f.write(string)
        with open(f"{dirstring}/eps_max_t_errors.txt", 'w') as f:
            for i in range(num_samples):
                string = f"{max_eps_field[i, 0]}"
                for j in range(settings['transfersteps']-1):
                    string += f" {max_eps_field[i, j+1]}"
                string += f"\n"
                f.write(string)
        with open(f"{dirstring}/sig_t_errors.txt", 'w') as f:
            for i in range(num_samples):
                string = f"{mean_sig_field[i, 0]}"
                for j in range(settings['transfersteps']-1):
                    string += f" {mean_sig_field[i, j+1]}"
                string += f"\n"
                f.write(string)
        with open(f"{dirstring}/sig_max_t_errors.txt", 'w') as f:
            for i in range(num_samples):
                string = f"{max_sig_field[i, 0]}"
                for j in range(settings['transfersteps']-1):
                    string += f" {max_sig_field[i, j+1]}"
                string += f"\n"
                f.write(string)
        with open(f"{dirstring}/homsig_t_errors.txt", 'w') as f:
            for i in range(num_samples):
                string = f"{mean_homsig[i, 0]}"
                for j in range(settings['transfersteps']-1):
                    string += f" {mean_homsig[i, j+1]}"
                string += f"\n"
                f.write(string)
        with open(f"{dirstring}/homsig_max_t_errors.txt", 'w') as f:
            for i in range(num_samples):
                string = f"{max_homsig[i, 0]}"
                for j in range(settings['transfersteps']-1):
                    string += f" {max_homsig[i, j+1]}"
                string += f"\n"
                f.write(string)

        # Plotting
        print(f"Plotting timestep errors")
        plot_t_error(mean_eps_field, settings, 'eps_Average', f'Average ' + r'$ ||\hat{\varepsilon}^{\omega}-\varepsilon^{\omega}|| $')
        plot_t_error(max_eps_field, settings, 'eps_Max', f'Max ' + r'$ ||\hat{\varepsilon}^{\omega}-\varepsilon^{\omega}|| $')
        plot_t_error(mean_sig_field, settings, 'sig_Average', f'Average ' + r'$ ||\hat{\sigma}^{\omega}-\sigma^{\omega}|| $')
        plot_t_error(max_sig_field, settings, 'sig_Max', f'Max ' + r'$ ||\hat{\sigma}^{\omega}-\sigma^{\omega}|| $')
        plot_t_error(mean_homsig, settings, 'sig_hom_Average', f'Average ' + r'$ ||\hat{\sigma}^{\Omega}-\sigma^{\Omega}|| $')
        plot_t_error(max_homsig, settings, 'sig_hom_Max', f'Max ' + r'$ ||\hat{\sigma}^{\Omega}-\sigma^{\Omega}|| $')


if test_type == 'comptime':
    FE_time_model = best_model.FE2_steps

    # Warmup rounds untimed
    for i, data in enumerate(plot_transf_dataLoad):
        data = data.clone().to(device)
        hom_stress, stiffness = FE_time_model(data)
        if i == 5:
            break
    best_model.matcomputetime = 0

    # Compute the total time
    start_time = time.time()

    for i, data in enumerate(plot_transf_dataLoad):
        data = data.clone().to(device)

        hom_stress, stiffness = FE_time_model(data)
        # Note that the stiffness is from normalized strain to unnormalized stress. Needs to be multiplied with normalization factor if used.
    tot_time = time.time() - start_time

    with open(f"{dirstring}/combined_comptime.txt", 'w') as f:
        f.write(f"stress_stiff_tot_time {tot_time}\n")
        f.write(f"stress_stiff_mat_time {best_model.matcomputetime}\n")

    # Then do only homstress
    with torch.no_grad():
        # Reset time for matcomputetime
        best_model.matcomputetime = 0
        if batch_size_comptime == 1:
            FE_time_model = best_model.FE2_steps_nostiff
        else:
            print("Stress multi batch mode not supported")
            exit()

        # Compute the total time
        hom_stresses = torch.empty((len(plot_transf_dataLoad), settings['transfersteps'], 3))
        start_time = time.time()
        for i, data in enumerate(plot_transf_dataLoad):
            data = data.clone().to(device)
            hom_stresses[i] = FE_time_model(data)
        tot_time = time.time() - start_time

    with open(f"{dirstring}/combined_comptime.txt", 'a') as f:
        f.write(f"stress_tot_time {tot_time}\n")
        f.write(f"stress_mat_time {best_model.matcomputetime}\n")

    # Print predicted homstresses
    with open(f"{dirstring}/pred_homstress.txt", 'w') as f:
        for i in range(len(plot_transf_dataLoad)):
            string = f"{hom_stresses[i, 0, 0]} {hom_stresses[i, 0, 1]} {hom_stresses[i, 0, 2]}"
            for j in range(settings['transfersteps']-1):
                string += f" {hom_stresses[i, j+1, 0]} {hom_stresses[i, j+1, 1]} {hom_stresses[i, j+1, 2]}"
            string += f"\n"
            f.write(string)

    print("Succesfully computed time.")
    exit()


"""
Plotting
"""

print_exp_config(f"{dirstring}/exp_config", configuration)

plot_field_unroll = settings['validationend'] != settings['transferend'] # and not settings['transfertrain']

plot_field_unroll_hybrid = plot_field_unroll
# Plot transfer dataset (not based on plot_cases, but on transferend)
if plot_field_unroll_hybrid:
    with torch.no_grad():
        print("### Plotting full & hybrid multistep results on the validation & transfer dataset ###")

        plot_data_arr = []
        m_arr = []

        # Add training samples
        train_iterator = iter(plot_train_dataLoad)
        for i in range(settings['trainend']):
            a = next(train_iterator)
            if i in configuration['plot_train']:
                plot_data_arr.append(a)
                m_arr.append(i)

        # Add validation samples
        val_iterator = iter(plot_val_dataLoad)
        for i in range(settings['validationend'] - settings['trainend'] ):
            a = next(val_iterator)
            if i in configuration['plot_val']:
                plot_data_arr.append(a)
                m_arr.append(i + settings['trainend'] )

        # Add extrapolate samples
        transf_iterator = iter(plot_transf_dataLoad)
        for i in range(settings['transferend'] - settings['validationend'] ):
            a = next(transf_iterator)
            if i in configuration['plot_extra']:
                plot_data_arr.append(a)
                m_arr.append(i + settings['validationend'] )

        for i in range(len(plot_data_arr)):
            m = m_arr[i]

            if m < settings['validationend']:
                macro_vec = macro_vec_train
                homstress = homstress_train
                index_macro_vec = m
                timesteps = settings['steps_ahead']
            else:
                macro_vec = macro_vec_transfer
                homstress = homstress_transfer
                index_macro_vec = m - settings['validationend']
                timesteps = settings['transfersteps']

            data = plot_data_arr[i].clone().to(device)

            start_pred_time = time.time()
            # Model predictions
            # Data.macro and macro_vec[m] have same values.
            pred_strains, hom_stress, pred_stress, pred_eps_p_eqs = best_model(data)
            # norm, norm, unnorm, unnorm

            end_pred_time = time.time()

            print(f"Time spent predicting: {end_pred_time - start_pred_time}")

            # Denormalize
            pred_strains = elemStrainNorm.denormalize(pred_strains.detach())
            hom_stress = homStressNorm.denormalize(hom_stress)
            targets_arr = data.y.cpu().numpy()
            targets_arr = elemStrainNorm.denormalize(targets_arr)

            # Very dirty, converting form (#Nodes x t x 3) to (t x #Nodes x 3) and to lists
            epspeq_all_steps = []
            targets_all_steps = []

            if pred_strains.dim() == 2:  # when having 1 timestep, the second dimension (t) is squeezed
                pred_strains = pred_strains.reshape(-1, 1, 3)

            pred_strains_arr = torch.empty((pred_strains.shape[1], pred_strains.shape[0], pred_strains.shape[2]))
            pred_stress_arr = torch.empty_like(pred_strains_arr)
            pred_eps_p_eqs_arr = torch.empty((pred_eps_p_eqs.shape[1], pred_eps_p_eqs.shape[0]))

            # Obtain targets
            for t in range(timesteps):
                targets_all_steps.append(targets_arr[:, t, :])
                pred_strains_arr[t] = pred_strains[:, t]
                pred_stress_arr[t] = pred_stress[:, t]
                pred_eps_p_eqs_arr[t] = pred_eps_p_eqs[:, t]

                if settings['epspeq_feature']:
                    if t == 0:
                        epspeq_t = data.x[:, 6]
                    else:
                        epspeq_t = data.epspeq_after.cpu().numpy()[:, t - 1]
                    epspeq_t = elemepspeqNorm.denormalize(epspeq_t)
                    epspeq_all_steps.append(epspeq_t)
                else:
                    epspeq_all_steps = None

            # Macroscopic
            macro_all_steps = torch.tensor(macro_vec)[index_macro_vec, :]
            macrostrain = macroStrainNorm.denormalize(macro_all_steps)

            # Plotting
            rve_unroll_visual(dirstring, pred_strains_arr, pred_stress_arr.detach(), pred_eps_p_eqs_arr.detach(), normalized_stresses, elemStressNorm, m, timesteps, mesh_ipArea_set[m], macrostrain, homstress[index_macro_vec], homStressNorm, targets_all_steps, epspeq_all_steps, mesh_node_coords_set, mesh_elem_nodes_set, settings, configuration['plot_finalonly'], extra_title='', homstress_pred = hom_stress[0].detach())

