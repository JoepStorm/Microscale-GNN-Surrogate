"""
Functions used to evaluate the GNN model and compute the loss function.

computeLoss is used in training, and computes the MSE over each component.

Two variations are made which do not average over all timesteps, but allow us to measure how the error evolves over the timesteps:
- computeLoss_t Default
- computeNorm_t Alternative loss that uses the norm instead of MSE. There are some subtle differences w.r.t. averaging over components

Losses are weighted sums of separate losses with their xi values:
- full-field strain
- full-field stress
- homogenized_stress  (legacy code)
After initial tests, we kept the xi for homogenized_stress at 0, and made sure the others always sum to one.

Normalized and unnormalized losses are computed.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

def computeLoss(model, data, settings, num_batches, device, elemStrainNorm, elemStressNorm, homStressNorm, store_homstress=False):
    data = data.to(device)

    strain_pred, hom_stress_pred, stress_pred, _ = model(data)
    strain_field_loss = F.mse_loss(strain_pred, data.y)
    stress_pred = elemStressNorm.normalize(stress_pred)
    stress_field_loss = F.mse_loss(stress_pred, data.y_stress)

    hom_loss = F.mse_loss(hom_stress_pred, data.target_homstress)

    # Loss includes homogenized loss, each with individual xi_.. value. In practice, we keep xi_stresshom = 0, and xi_strainfield + xi_stressfield = 1
    loss = settings['xi_stresshom'] * torch.sqrt(hom_loss) + settings['xi_strainfield'] * torch.sqrt(strain_field_loss) + settings['xi_stressfield'] * torch.sqrt(stress_field_loss)

    # Make independent of batch size and detach
    strain_field_loss = strain_field_loss.detach().item()
    stress_field_loss = stress_field_loss.detach().item()
    hom_loss = hom_loss.detach().item()

    unnorm_strain_field_loss = strain_field_loss * elemStrainNorm.factor ** 2
    unnorm_stress_field_loss = stress_field_loss * elemStressNorm.factor ** 2
    unnorm_hom_loss = hom_loss * homStressNorm.factor ** 2

    # return loss, field_loss, hom_loss     # V1
    if not store_homstress:    # default
        return loss, unnorm_strain_field_loss, unnorm_stress_field_loss, unnorm_hom_loss, strain_field_loss, stress_field_loss, hom_loss      # V2
    else:
        return loss, unnorm_strain_field_loss, unnorm_stress_field_loss, unnorm_hom_loss, strain_field_loss, stress_field_loss, hom_loss, homStressNorm.denormalize(hom_stress_pred), homStressNorm.denormalize(data.target_homstress)


# Version used in load model:
def computeLoss_t(model, data, device, elemStrainNorm, elemStressNorm, homStressNorm, comp_baseline=False):
    data = data.to(device)

    # Denorm targets
    unnorm_strain_target = elemStrainNorm.denormalize(data.y)
    unnorm_hom_stress_target = homStressNorm.denormalize(data.target_homstress)
    unnorm_stress_target = elemStressNorm.denormalize(data.y_stress)

    if not comp_baseline:       # actual predictions
        strain_pred, hom_stress_pred, unnorm_stress_pred, _ = model(data)

        # Denorm preds
        unnorm_strain_pred = elemStrainNorm.denormalize(strain_pred)
        unnorm_hom_stress_pred = homStressNorm.denormalize(hom_stress_pred)
        # stress_pred already unnormalized

    else:       # predict all zeros
        unnorm_strain_pred = torch.zeros_like(data.y)
        unnorm_hom_stress_pred = torch.zeros_like(data.target_homstress)
        unnorm_stress_pred = torch.zeros_like(data.y_stress)

    ##### Compute err but don't sum over timesteps #####

    ## Strains
    # MAE  -   note that use non-squared factor.
    mae_reduced = nn.L1Loss(reduction='none')
    strain_field_loss_t = mae_reduced(unnorm_strain_pred, unnorm_strain_target)
    # Mean
    # Loop over elements, then over components
    eps_field_loss_t = torch.mean(torch.mean(strain_field_loss_t, dim=0), dim=1).detach().cpu().numpy()
    # Max
    eps_max_field_loss_t = torch.amax(torch.amax(strain_field_loss_t, dim=0), dim=1).detach().cpu().numpy()

    ### Stress field
    stress_field_loss_t = mae_reduced(unnorm_stress_pred, unnorm_stress_target)
    # Mean
    sig_field_loss_t = torch.mean(torch.mean(stress_field_loss_t, dim=0), dim=1).detach().cpu().numpy()
    # Max
    sig_max_field_loss_t = torch.amax(torch.amax(stress_field_loss_t, dim=0), dim=1).detach().cpu().numpy()

    ### Hom stress
    # here we loop over batch, then over components. (Even though batch size should be 1)
    hom_loss_t = mae_reduced(unnorm_hom_stress_pred, unnorm_hom_stress_target)
    # Mean
    homsig_loss_t = torch.mean(torch.mean(hom_loss_t, dim=0), dim=1).detach().cpu().numpy()
    # Max
    homsig_max_loss_t = torch.amax(torch.amax(hom_loss_t, dim=0), dim=1).detach().cpu().numpy()

    return eps_field_loss_t, eps_max_field_loss_t, sig_field_loss_t, sig_max_field_loss_t, homsig_loss_t, homsig_max_loss_t  # V2

# Compute the differences between the norms of the predictions and the targets
# Additionally, the baseline for the strains is the average of the input strains.
def computeNorm_t(model, data, device, elemStrainNorm, elemStressNorm, homStressNorm, comp_baseline=False, macroStrainNorm=None):
    data = data.to(device)

    # Denorm targets
    unnorm_strain_target = elemStrainNorm.denormalize(data.y)
    unnorm_hom_stress_target = homStressNorm.denormalize(data.target_homstress)
    unnorm_stress_target = elemStressNorm.denormalize(data.y_stress)

    if not comp_baseline:       # actual predictions
        strain_pred, hom_stress_pred, unnorm_stress_pred, _ = model(data)

        # Denorm preds
        unnorm_strain_pred = elemStrainNorm.denormalize(strain_pred)
        unnorm_hom_stress_pred = homStressNorm.denormalize(hom_stress_pred)
        # stress_pred already unnormalized
    else:       # predict all zeros
        unnorm_strain_pred = torch.zeros_like(data.y)
        unnorm_hom_stress_pred = torch.zeros_like(data.target_homstress)
        unnorm_stress_pred = torch.zeros_like(data.y_stress)

    if macroStrainNorm is not None:   # Macro strain as baseline for strains only!
        # Broadcast the macroscopic strain to each element as an alternative baseline
        macro_0 = macroStrainNorm.denormalize(data.x[0,3:6])
        macro_rest = macroStrainNorm.denormalize(data.macro[0, :])
        macro_strain =  torch.cat((macro_0.view(1,-1), macro_rest), dim=0)
        unnorm_strain_pred = torch.broadcast_to(macro_strain, unnorm_strain_target.shape)


    ##### Compute err but don't sum over timesteps #####

    ## Strains
    # MAE  -   note that use non-squared factor.
    # mae_reduced = nn.L1Loss(reduction='none')
    # strain_field_loss_t = mae_reduced(unnorm_strain_pred, unnorm_strain_target)

    print(f"Note: computing norm of loss, not absolute!")
    mse_reduced = nn.MSELoss(reduction='none')

    strain_field_loss_t = mse_reduced(unnorm_strain_pred, unnorm_strain_target)
    # Sum over components**2 then sqrt total to get norms
    strain_field_norms = torch.sqrt(torch.sum(strain_field_loss_t, dim=2))
    # Mean over elements
    eps_field_loss_t = torch.mean(strain_field_norms, dim=0).detach().cpu().numpy()
    # Max
    eps_max_field_loss_t = torch.amax(strain_field_norms, dim=0).detach().cpu().numpy()

    ### Stress field
    stress_field_loss_t = mse_reduced(unnorm_stress_pred, unnorm_stress_target)
    # Mean
    stress_field_norms = torch.sqrt(torch.sum(stress_field_loss_t, dim=2))

    sig_field_loss_t = torch.mean(stress_field_norms, dim=0).detach().cpu().numpy()
    # Max
    sig_max_field_loss_t = torch.amax(stress_field_norms, dim=0).detach().cpu().numpy()

    ### Hom stress
    # here we loop over batch, then over components. (Even though batch size should be 1)
    hom_loss_t = mse_reduced(unnorm_hom_stress_pred, unnorm_hom_stress_target)
    homstress_norms = torch.sqrt(torch.sum(hom_loss_t, dim=2))
    # Mean
    homsig_loss_t = torch.mean(homstress_norms, dim=0).detach().cpu().numpy()
    # Max
    homsig_max_loss_t = torch.amax(homstress_norms, dim=0).detach().cpu().numpy()

    return eps_field_loss_t, eps_max_field_loss_t, sig_field_loss_t, sig_max_field_loss_t, homsig_loss_t, homsig_max_loss_t  # V2
