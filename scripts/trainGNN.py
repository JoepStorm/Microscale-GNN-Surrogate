"""
Script to train a GNN model.
Usage: python3 trainGNN.py [batch_job_id]
where batch_job_id is an integer, used to select a specific set of hyperparameters from the settings file.

A folder is created in the results directory, where the trained model, losses, and the configuration file are stored.
A few plots from the validation set will also be created.

The settings in ConfigGNN are set to values that lead to a quick training run for development, not the actual values used in the paper.

In the settings, 'trim_for_test' is set to True, this only uses the first 20 samples for experimentation during development.
To run the full dataset, set 'trim_for_test' to False.

Plotting is done using a custom version of the SciencePlots style: https://github.com/garrettj403/SciencePlots.
If this is not installed, remove the following 2 lines from all plotting scripts:
rc('text', usetex=True)
plt.style.use(['science', 'bright'])

A transfer set must be specified, even if it is not used.
"""
import os
import sys
import copy
import time
import numpy as np
import torch_geometric
import torch
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from GNN_MPL import GNN
from readConfig import read_config, get_batch_settings, printConfig
from plotUtils import plot_losses, rve_unroll_visual, plot_sep_losses
from dirUtils import save_dir, make_dir
from load_data import load_data
from lossFunc import computeLoss
from graphDatasets import createDataset
import normUtils

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
Inputs
"""
start = time.time()

# Read the configuration
settings_file = '../fem/ConfigGNN'
settings = read_config(settings_file)

# See if the script is passed with additional arguments, in which case it is a batch script and additional settings are loaded
batch_job = False
sys_args = sys.argv
if len(sys_args) > 1:       # There is a parameter given, so part of batch job.
    batch_job = int(sys_args[1])
    settings = get_batch_settings(settings, batch_job)

assert settings['xi_strainfield']+settings['xi_stressfield']+settings['xi_stresshom'] == 1, f"xi values should sum to 1, but are {settings['xi_strainfield']}, {settings['xi_stressfield']} and {settings['xi_stresshom']} respectively."

# Make directory where settings and results are stored, this is renamed at the end
tmp_foldername = make_dir(settings['saveresultdir'], batch_job)

# Store current configuration
printConfig(f'{tmp_foldername}/ConfigGNN', settings)

# Trim dataset size when testing for faster results
if settings['trim_for_test']:
    settings['max_epochs'] = 20
    settings['trainsize'] = 10
    settings['trainend'] = 10
    settings['validationend'] = 20
    settings['transferend'] = 20

np.random.seed(settings['seed'])

### Obtain data

normclassexists = False
norm_scalers = []
geomFeatNorm = None
edgeFeatNorm = None
if settings['pretrain'] != 'None':
    # Get normalization parameters from pretrained model
    norm_scalers = normUtils.readNormFile(f'{settings["pretrain"]}/normbounds')
    geomFeatNorm, edgeFeatNorm, _, _, _, _, _, _ = norm_scalers
    normclassexists = True

fibers, macro_vec_train, macro_vec_transfer, eps_set, eps_set_noise, epspeq_set_noise, macroStrainNorm, elemStrainNorm, elemepspNorm, elemepspeqNorm, homstress_train, homstress_transfer, homStressNorm, normalized_stresses, elemStressNorm = load_data(settings, norm_scalers, settings['trim_for_test'])

if settings['stress_as_input_field']:
    elemStrainNorm.donorm = elemStressNorm.donorm
    elemStrainNorm.factor = elemStressNorm.factor
    elemStrainNorm.normmean = elemStressNorm.normmean
    elemStrainNorm.normstd = elemStressNorm.normstd

faulty_meshes = []

"""
Create individual samples, matching data with graphs
"""

trainfibnum = settings['fibers_considered'][0]
print(f"Fibers: {trainfibnum}, with meshes from meshdir: {settings['meshdir']} ")

strain_components = 3  # (eps_x, eps_y, eps_xy)
pred_components = 3  # (eps_x, eps_y, eps_xy)

geom_features = trainfibnum * 2
base_inputfeats = geom_features
# Not a clean implementation
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
Initialize graphs for various meshes
"""
datasets, mesh_elem_nodes_set, mesh_node_coords_set, mesh_ipArea_set, geomFeatNorm, edgeFeatNorm = createDataset(settings, init_tensor_dtype, macro_vec_train, homstress_train, macro_vec_transfer, homstress_transfer, fibers, trainfibnum, strain_components, pred_components, elemStrainNorm, elemepspeqNorm, eps_set, epspeq_set_noise, geom_features, normalized_stresses, normclassexists, geomFeatNorm, edgeFeatNorm)

print("Graphs created")
# Deleting unnecessary data to free up memory
del(eps_set)
del(eps_set_noise)
del(epspeq_set_noise)


"""
Set-up dataloader
"""


# Sampling
# Select subset_indices
if settings['random_trainset']:
    print(f"Selecting a random subset for the training data with seed {settings['seed']}")
    subset_indices = np.random.choice(settings['trainend'], settings['trainsize'], replace=False)
else:
    print(f"Selecting a fixed subset for the training data in range {settings['trainend'] - settings['trainsize']} - {settings['trainend']}")
    subset_indices = np.arange(settings['trainend'] - settings['trainsize'], settings['trainend'])
trainset = torch.utils.data.Subset(datasets, subset_indices)

# Load dataloaders
print("Datasets created. Creating dataloaders..")
torch_geometric.seed_everything(settings['seed'])

train_dataLoad = DataLoader(trainset, batch_size=settings['batch_size'], shuffle=True, num_workers=0, pin_memory=False)

val_dataLoad   = DataLoader(datasets[settings['trainend']:settings['validationend']], batch_size=settings['batch_size'], shuffle=False, num_workers=0, pin_memory=False)
transf_dataLoad = None
if settings['transferend'] != settings['validationend']:
    print(f"Transfer: from {settings['validationend']} to {settings['transferend']}")
    transf_dataLoad   = DataLoader(datasets[settings['validationend']:settings['transferend']], batch_size=settings['batch_size'], shuffle=False, num_workers=0, pin_memory=False)

# For plotting we create additional datasets. Repeat above, but for training size = 1 and no shuffle in training
plot_train_dataLoad = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
plot_val_dataLoad   = DataLoader(datasets[settings['trainend']:settings['validationend']], batch_size=1, shuffle=False, num_workers=0)

if settings['transferend'] != settings['validationend']:
    plot_transf_dataLoad = DataLoader(datasets[settings['validationend']:settings['transferend']], batch_size=1, shuffle=False, num_workers=0)


"""
Train networks
"""
# Select the activation function
act = settings['activation_func']
if  act == 'relu' or act == 'leakyrelu':
    settings['w_init'] = 'kaiming_uniform'
elif act == 'selu' or act == 'silu':
    settings['w_init'] = None
else:
    settings['w_init'] = 'glorot'

# We use the number of layers from the first element in the list. The list was only used during development for comparing various sizes.
layers = settings['layers_arr'][0]
print(f"L:{layers} HN:{settings['hidden_nodes']} act:{settings['activation_func']}")

"""
Initializing class
"""
print(f"Initializing class..")
model = GNN(num_features, pred_features, layers, settings, device, elemStrainNorm, elemepspeqNorm, homStressNorm, elemStressNorm).to(device)

if settings['SGD_optim']:
    optimizer = torch.optim.SGD(model.parameters(), lr=settings['learn_rate'], weight_decay=settings['weight_decay'])
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learn_rate'], weight_decay=settings['weight_decay'])

# Load a pre-trained model
if settings['pretrain'] != 'None':
    model.load_state_dict(torch.load(settings['pretrain'] + "/model", map_location=device))

# Copy model to best_model
best_model = copy.deepcopy(model)

"""
Learning rate scheduler
"""
# Hardcoded
if settings['adaptive_lr']:
    warmup_scheduler = LinearLR(optimizer, start_factor=0.05, end_factor=1, total_iters=10)
    decay_scheduler = ExponentialLR(optimizer, gamma=0.998)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[10])

"""
Evaluating
"""

loss_arr = np.zeros(settings['max_epochs'])
val_loss_arr = np.zeros(settings['max_epochs'])
transfer_loss_arr = np.zeros(settings['max_epochs'])

sep_loss_arr = np.zeros((settings['max_epochs'], 3))
sep_val_loss_arr = np.zeros((settings['max_epochs'], 3))
sep_transfer_loss_arr = np.zeros((settings['max_epochs'], 3))
sep_loss_arr_norm = np.zeros((settings['max_epochs'], 3))
sep_val_loss_arr_norm = np.zeros((settings['max_epochs'], 3))
sep_transfer_loss_arr_norm = np.zeros((settings['max_epochs'], 3))

best_MSE_val = 1e9
best_train_MSE = 1e9
best_VAL_MSE = 1e9
best_transf_MSE = 0.0

print(f"Starting main training loop..")
start = time.time()
iterator = tqdm(range(settings['max_epochs']), desc='', leave=True)
settings['abort_training'] = False
settings['skip_minibatch'] = False

for epoch in iterator:
    """ -------------------- Training -----------------------   """
    ep_loss = 0
    model.train()
    batches_skipped = 0

    for i, data in enumerate(train_dataLoad):
        # Computeloss also calls the forward pass.
        loss, strain_field_loss, stress_field_loss, hom_loss, strain_field_loss_norm, stress_field_loss_norm, hom_loss_norm = computeLoss(model, data, settings, len(train_dataLoad), device, elemStrainNorm, elemStressNorm, homStressNorm)

        if settings['skip_minibatch'] == True:
            print("Minibatch skipped in training.")
            batches_skipped += 1
            settings['skip_minibatch'] = False
            continue

        ep_loss += float(loss*loss)  # square to store same values for comparison
        sep_loss_arr[epoch,0] += strain_field_loss
        sep_loss_arr[epoch,1] += stress_field_loss
        sep_loss_arr[epoch,2] += hom_loss
        sep_loss_arr_norm[epoch,0] += strain_field_loss_norm
        sep_loss_arr_norm[epoch,1] += stress_field_loss_norm
        sep_loss_arr_norm[epoch,2] += hom_loss_norm

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    loss_arr[epoch] = np.sqrt(ep_loss / len(train_dataLoad))

    if settings['adaptive_lr']:
        lr_scheduler.step()

    """ -------------------- Validation -----------------------   """
    if settings['main_error_train']:  # Otherwise error in first epoch for best val mse when there are 0 validation samples.
        val_loss = 0
    if epoch % settings['val_frequency'] == 0 and len(val_dataLoad) > 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_dataLoad):
                loss, strain_field_loss, stress_field_loss, hom_loss, strain_field_loss_norm, stress_field_loss_norm, hom_loss_norm = computeLoss(model, data, settings, len(val_dataLoad), device, elemStrainNorm, elemStressNorm, homStressNorm)

                if settings['skip_minibatch'] == True:
                    print("Minibatch skipped in validation.")
                    batches_skipped += 1
                    settings['skip_minibatch'] = False
                    continue

                val_loss += float(loss * loss)  # square to store same values for comparison
                sep_val_loss_arr[epoch, 0] += strain_field_loss
                sep_val_loss_arr[epoch, 1] += stress_field_loss
                sep_val_loss_arr[epoch, 2] += hom_loss
                sep_val_loss_arr_norm[epoch, 0] += strain_field_loss_norm
                sep_val_loss_arr_norm[epoch, 1] += stress_field_loss_norm
                sep_val_loss_arr_norm[epoch, 2] += hom_loss_norm

            val_loss_arr[epoch] = np.sqrt(val_loss / len(val_dataLoad))


    """ -------------------- Transfer -----------------------"""
    if epoch % settings['transf_frequency'] == 0:
        model.eval()
        transf_loss = 0
        if settings['transferend'] != settings['validationend'] and settings['transfertrain']:
            with torch.no_grad():
                for i, data in enumerate(transf_dataLoad):
                    if i == 0 and settings['storetransfhom']:
                        loss, strain_field_loss, stress_field_loss, hom_loss, strain_field_loss_norm, stress_field_loss_norm, hom_loss_norm, hom_pred, hom_true = computeLoss(model, data, settings, len(transf_dataLoad), device, elemStrainNorm, elemStressNorm, homStressNorm, store_homstress=True)
                    else:
                        loss, strain_field_loss, stress_field_loss, hom_loss, strain_field_loss_norm, stress_field_loss_norm, hom_loss_norm = computeLoss(model, data, settings, len(transf_dataLoad), device, elemStrainNorm, elemStressNorm, homStressNorm)

                    if settings['skip_minibatch'] == True:
                        print("Minibatch skipped in Transfer.")
                        batches_skipped += 1
                        settings['skip_minibatch'] = False
                        continue

                    transf_loss += float(loss * loss)  # square to store same values for comparison
                    sep_transfer_loss_arr[epoch, 0] += strain_field_loss
                    sep_transfer_loss_arr[epoch, 1] += stress_field_loss
                    sep_transfer_loss_arr[epoch, 2] += hom_loss
                    sep_transfer_loss_arr_norm[epoch, 0] += strain_field_loss_norm
                    sep_transfer_loss_arr_norm[epoch, 1] += stress_field_loss_norm
                    sep_transfer_loss_arr_norm[epoch, 2] += hom_loss_norm

                    if i == 0 and settings['storetransfhom']:  # Store the homogenized stress output during training
                        if epoch == 0:  # Write true values on first line
                            with open(f'{tmp_foldername}/transfer_hom_stress', 'a') as f:
                                f.write(' '.join(str(i.item()) for i in hom_true.flatten()) + '\n')
                        with open(f'{tmp_foldername}/transfer_hom_stress', 'a') as f:
                            f.write(' '.join(str(i.item()) for i in hom_pred.flatten()) + '\n')

                transfer_loss_arr[epoch] = np.sqrt(transf_loss / len(transf_dataLoad))

    if batches_skipped > 100:
        settings['abort_training'] = True

    if settings['abort_training']:
        print("Aborting training...")
        break

    # Save model with best validation accuracy:
    if settings['main_error_train']:
        main_loss = ep_loss
    else:
        main_loss = val_loss

    if main_loss < best_MSE_val:
        best_model.load_state_dict(model.state_dict())
        best_MSE_val = main_loss
        best_train_MSE = ep_loss
        best_VAL_MSE = val_loss
        best_epoch = epoch
        if settings['transferend'] != settings['validationend'] and settings['transfertrain']:
            best_transf_MSE = transf_loss

    if epoch > best_epoch + settings['early_stop_epochs'] and epoch > settings['min_epochs']:
        break

    elapsed_time = time.time() - start
    if elapsed_time > settings['max_train_hours'] * 60 * 60:
        print(f"Maximum training time of {settings['max_train_hours']} hours reached!")
        break

    remainder_update_epochs = epoch % settings['update_epochs']
    iterator.set_description(f"Tr:{ loss_arr[epoch]:.5f}, Vl:{ val_loss_arr[epoch - remainder_update_epochs]:.5f}, Trns:{transfer_loss_arr[epoch - remainder_update_epochs]:.5f} Skipped: {batches_skipped}")
end = time.time()
print(f"Total training time: {end - start} seconds")

# Divide seperate losses to account for batch size differences
sep_loss_arr /= len(train_dataLoad)
sep_loss_arr_norm /= len(train_dataLoad)
if len(val_dataLoad) > 0:
    sep_val_loss_arr /= len(val_dataLoad)
    sep_val_loss_arr_norm /= len(val_dataLoad)
if settings['transferend'] != settings['validationend']:
    sep_transfer_loss_arr /= len(transf_dataLoad)
    sep_transfer_loss_arr_norm /= len(transf_dataLoad)

# Convert separate losses to RMSE
sep_loss_arr = np.sqrt(sep_loss_arr[:epoch+1,:])
sep_val_loss_arr = np.sqrt(sep_val_loss_arr[:epoch+1,:])
sep_transfer_loss_arr = np.sqrt(sep_transfer_loss_arr[:epoch+1,:])
sep_loss_arr_norm = np.sqrt(sep_loss_arr_norm[:epoch+1,:])
sep_val_loss_arr_norm = np.sqrt(sep_val_loss_arr_norm[:epoch+1,:])
sep_transfer_loss_arr_norm = np.sqrt(sep_transfer_loss_arr_norm[:epoch+1,:])

# Set up best model
best_model.train(mode=False)
best_model.settings['run_material'] = True      # If material model not used, now do use it for plotting

# Cut the losses to latest epoch
loss_arr = loss_arr[:epoch]
val_loss_arr = val_loss_arr[:epoch]
transfer_loss_arr = transfer_loss_arr[:epoch]

print('Final val RMSE: ', val_loss_arr[-1].item(), ' Best validation RMSE: ', np.sqrt(best_VAL_MSE),' at epoch ', best_epoch)


"""
Store result
"""
losses = [loss_arr, val_loss_arr, transfer_loss_arr]
loss_names = ['Training', 'Validation', 'Transfer']

# Store trained model:
if settings['savemodel']:
    torch.save(best_model.state_dict(), tmp_foldername + '/model')

# Store normalization bounds in file
normUtils.writeNormFile(f"{tmp_foldername}/normbounds", settings, settings['edge_vec'], geomFeatNorm, edgeFeatNorm, macroStrainNorm, elemStrainNorm, elemepspNorm, elemepspeqNorm, homStressNorm, elemStressNorm)

# Store loss arrays in file
with open(f"{tmp_foldername}/losses", 'w') as f:
    if settings['transferend'] != settings['validationend'] and settings['transfertrain']:
        for l, vl, tl in zip(loss_arr, val_loss_arr, transfer_loss_arr):
            f.write(f"{l} {vl} {tl}\n")
    else:
        for l, vl in zip(loss_arr, val_loss_arr):
            f.write(f"{l} {vl}\n")

# Store separate loss arrays in file
with open(f"{tmp_foldername}/sep_losses", 'w') as f:
    if settings['transferend'] != settings['validationend'] and settings['transfertrain']:
        for l1, l2, l3, vl1, vl2, vl3, tl1, tl2, tl3 in zip(sep_loss_arr[:,0], sep_loss_arr[:,1], sep_loss_arr[:,2], sep_val_loss_arr[:,0], sep_val_loss_arr[:,1], sep_val_loss_arr[:,2], sep_transfer_loss_arr[:,0], sep_transfer_loss_arr[:,1], sep_transfer_loss_arr[:,2]):
            f.write(f"{l1} {l2} {l3} {vl1} {vl2} {vl3} {tl1} {tl2} {tl3}\n")
    else:
        for l1, l2, l3, vl1, vl2, vl3 in zip(sep_loss_arr[:,0], sep_loss_arr[:,1], sep_loss_arr[:,2], sep_val_loss_arr[:,0], sep_val_loss_arr[:,1], sep_val_loss_arr[:,2]):
            f.write(f"{l1} {l2} {l3} {vl1} {vl2} {vl3}\n")

if settings['validationend'] != settings['transferend']:
    print("### Computing total transfer loss:")
    with torch.no_grad():
        best_trans_loss = 0
        best_trans_loss_strain = 0
        best_trans_loss_stress = 0
        best_trans_loss_hom = 0
        best_trans_loss_strain_norm = 0
        best_trans_loss_stress_norm = 0
        best_trans_loss_hom_norm = 0

        # Run full transfer dataset
        for i, data in enumerate(transf_dataLoad):
            loss, strain_field_loss, stress_field_loss, hom_loss, strain_field_loss_norm, stress_field_loss_norm, hom_loss_norm = computeLoss(best_model, data, settings, len(transf_dataLoad), device, elemStrainNorm, elemStressNorm, homStressNorm)
            best_trans_loss += loss.detach()
            best_trans_loss_strain += strain_field_loss
            best_trans_loss_stress += stress_field_loss
            best_trans_loss_hom += hom_loss
            best_trans_loss_strain_norm += strain_field_loss_norm
            best_trans_loss_stress_norm += stress_field_loss_norm
            best_trans_loss_hom_norm += hom_loss_norm

        # Store best in file
        with open(f"{tmp_foldername}/best_trans_losses", 'w') as f:
            f.write(f"{best_trans_loss} {best_trans_loss_strain} {best_trans_loss_stress} {best_trans_loss_hom} {best_trans_loss_strain_norm} {best_trans_loss_stress_norm} {best_trans_loss_hom_norm}")

print("### Plotting training RMSE ###")
# Plot Training RMSE
plot_losses(losses, loss_names, settings, tmp_foldername, best_epoch)
sep_losses_arr = [sep_loss_arr[:,0], sep_loss_arr[:,1], sep_loss_arr[:,2], sep_val_loss_arr[:,0], sep_val_loss_arr[:,1], sep_val_loss_arr[:,2], sep_transfer_loss_arr[:,0], sep_transfer_loss_arr[:,1], sep_transfer_loss_arr[:,2]]
sep_losses_ar_norm = [sep_loss_arr_norm[:,0], sep_loss_arr_norm[:,1], sep_loss_arr_norm[:,2], sep_val_loss_arr_norm[:,0], sep_val_loss_arr_norm[:,1], sep_val_loss_arr_norm[:,2], sep_transfer_loss_arr_norm[:,0], sep_transfer_loss_arr_norm[:,1], sep_transfer_loss_arr_norm[:,2]]
plot_sep_losses(sep_losses_arr, sep_losses_ar_norm, settings, tmp_foldername, best_epoch)

"""
Plotting full-field samples
"""
try:
    plot_transf_dataLoad
except NameError:
    plot_transf_dataLoad = transf_dataLoad

unroll_folder = f"{tmp_foldername}/unroll"
os.mkdir(unroll_folder)

settings['plot_fullfield'] = True
final_only = True

# Plotting
with torch.no_grad():

    print("### Plotting full & hybrid multistep results on the validation & transfer dataset ###")
    if settings['validationend'] != settings['transferend']:
        plot_data_arr = [next(iter(plot_val_dataLoad)), next(iter(plot_transf_dataLoad))]   # first validation and first transfer sample
        m_arr = [settings['trainend'],settings['validationend']]  # This must correspond to the exact plot_data_arr selection!
    elif len(plot_val_dataLoad) > 0:
        plot_data_arr = [next(iter(plot_train_dataLoad)), next(iter(plot_val_dataLoad))]   # first train and first validation sample
        m_arr = [settings['trainend']-settings['trainsize'],settings['trainend']]  # This must correspond to the exact plot_data_arr selection!
    else:
        plot_data_arr = [next(iter(plot_train_dataLoad))]  # first train sample
        m_arr = [settings['trainend'] - settings['trainsize']]  # This must correspond to the exact plot_data_arr selection!

    for i in range(len(plot_data_arr)):
        m = m_arr[i]
        print(f"Plotting m: {m}")

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

        # Model predictions
        pred_strains, hom_stress, pred_stress, pred_eps_p_eqs = best_model(data)
        # norm, norm, unnorm, unnorm

        # Denormalize
        pred_strains = elemStrainNorm.denormalize(pred_strains.detach())
        hom_stress = homStressNorm.denormalize(hom_stress)

        targets_arr = data.y.cpu().numpy()
        targets_arr = elemStrainNorm.denormalize(targets_arr)

        # Very dirty, converting form (#Nodes x t x 3) to (t x #Nodes x 3) and to lists
        epspeq_all_steps = []
        targets_all_steps = []

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
        rve_unroll_visual(unroll_folder, pred_strains_arr, pred_stress_arr.detach(), pred_eps_p_eqs_arr.detach(), normalized_stresses, elemStressNorm, m, timesteps, mesh_ipArea_set[m], macrostrain, homstress[index_macro_vec], homStressNorm, targets_all_steps, epspeq_all_steps, mesh_node_coords_set, mesh_elem_nodes_set, settings, final_only, homstress_pred = hom_stress[0].detach())


# Rename temporary folder to informative name
datasize = len(val_dataLoad)
if settings['main_error_train']:
    datasize = len(train_dataLoad)

save_dir(tmp_foldername, sep_val_loss_arr[best_epoch, 2], np.sqrt(best_train_MSE / datasize), np.sqrt(best_VAL_MSE / datasize), np.sqrt(best_transf_MSE / datasize), settings, layers, trainfibnum, epoch, end - start, use_cuda, batch_job)

