"""
Script that loads the datasets (excluding mesh files).
Polars is used as an alternative to Pandas, as it is significantly faster and uses less memory.
The data formats are altered to support this. All large data files are expanded with dummy values such that all rows have the same number of columns

Legacy features make some parts significantly more complex than required.
"""
import numpy as np
import polars
import normUtils

"""
Inputs
"""
def load_data(settings, norm_scalers = [], trim_for_testing = False, empty_micro = False):
    print("Loading data...")

    norm_given = False
    if len(norm_scalers) != 0:
        print('Using existing normalization bounds.')
        norm_given = True

    fibers = polars.read_csv(settings['meshdir'] + '/fiber_coords_expanded', separator=',', has_header=False).to_numpy()
    fibers_set = []
    for fib_list in fibers:
        fibers_set.append(fib_list[~np.isnan(fib_list)].tolist())
    fibers_set = fibers_set[:settings['validationend']]     # Allows selecting from larger datafolders

    if settings['validationend'] != settings['transferend']:
        fibers_transfer = polars.read_csv(settings['transfermesh'] + '/fiber_coords_expanded', separator=',', has_header=False).to_numpy()
        for fib_list in fibers_transfer:
            fibers_set.append(fib_list[~np.isnan(fib_list)].tolist())

    if trim_for_testing:
        if settings['load_vec_separate']:
            df_load = polars.read_csv(settings['datafolder'] + '/macrostrain_trimmed.data', separator=' ', has_header=False)

        df_stress = polars.read_csv(settings['datafolder'] + '/macrostress_trimmed.data', separator=' ', has_header=False)
    else:
        if settings['load_vec_separate']:
            df_load = polars.read_csv(settings['datafolder'] + '/macrostrain.data', separator=' ', has_header=False)

        df_stress = polars.read_csv(settings['datafolder'] + '/macrostress.data', separator=' ', has_header=False)


    # Obtain normalization bounds if provided
    if norm_given:
        geomFeatNorm, edgeFeatNorm, macroStrainNorm, elemStrainNorm, elemepspNorm, elemepspeqNorm, homStressNorm, elemStressNorm = norm_scalers

    """
    Obtain dataset.
    """
    if settings['load_vec_as_feature']:
        if settings['load_vec_in_data']:
            print(f"load_vec_in_data no longer supported")
            exit()
            macro_vec_unnorm = df.iloc[:, 0:3].to_numpy().astype(float)
            macro_vec_unnorm = macro_vec_unnorm.reshape(-1, settings['timesteps'], 3)[:,:settings['steps_ahead']]

        elif settings['load_vec_separate']:
            macro_vec_unnorm = df_load.to_numpy().astype(float)
            # Reshape to (N, t_data, 3); then select (N, t_used, 3)
            macro_vec_unnorm = macro_vec_unnorm.reshape(-1, settings['timesteps'], 3)[:,:settings['steps_ahead']]

        if not norm_given:
            macroStrainNorm = normUtils.normUnitvar(macro_vec_unnorm, donorm=settings['norm_inputs'], normmean=0)

        macro_vec = macroStrainNorm.normalize(macro_vec_unnorm)
        if sum(np.isnan(macro_vec.reshape(-1))) != 0:
            print(f"macro_vec Tensor contains NaN")
            exit()

    print("Loading transfer data")
    macro_vec_transfer = []
    if settings['validationend'] != settings['transferend']:
        if empty_micro is False:
            # Load macro data
            df_load_transfer = polars.read_csv(settings['transferdata'] + '/macrostrain.data', separator=' ', has_header=False)
            df_stress_transfer = polars.read_csv(settings['transferdata'] + '/macrostress.data', separator=' ', has_header=False)
            macro_vec_unnorm_transfer = df_load_transfer.to_numpy().astype(float)
            # Reshape to (N, t_data, 3); then select (N, t_used, 3)
            macro_vec_unnorm_transfer = macro_vec_unnorm_transfer.reshape(-1, settings['transfersteps'], 3)[:,:settings['transfersteps']]
            macro_vec_transfer = macroStrainNorm.normalize(macro_vec_unnorm_transfer)

            # Load micro data
            df_eps_transfer = polars.read_csv(settings['transferdata'] + '/strains_expanded.data', has_header=False, separator=',').with_columns(polars.all().cast(polars.Float32, strict=False))
            df_sig_transfer = polars.read_csv(settings['transferdata'] + '/stress_expanded.data', has_header=False, separator=',').with_columns(polars.all().cast(polars.Float32, strict=False))

            # df_epsp_transfer = polars.read_csv(settings['transferdata'] + '/epsp_expanded.data', has_header=False, separator=',')
            df_epspeq_transfer = polars.read_csv(settings['transferdata'] + '/epspeq_expanded.data', has_header=False, separator=',').with_columns(polars.all().cast(polars.Float32, strict=False))

            # df_epsp_transfer = df_epsp_transfer.to_numpy()
            df_epspeq_transfer = df_epspeq_transfer.to_numpy()
            df_eps_transfer = df_eps_transfer.to_numpy()
            df_sig_transfer = df_sig_transfer.to_numpy()
        else:   # Only get real macro vector, rest create zero matrices
            df_load_transfer = polars.read_csv(settings['transferdata'] + '/macrostrain.data', separator=' ', has_header=False)
            macro_vec_unnorm_transfer = df_load_transfer.to_numpy().astype(float)
            macro_vec_unnorm_transfer = macro_vec_unnorm_transfer.reshape(-1, settings['transfersteps'], 3)[:,:settings['transfersteps']]
            macro_vec_transfer = macroStrainNorm.normalize(macro_vec_unnorm_transfer)

            df_epspeq_transfer = np.zeros((settings['transferend'] - settings['validationend'], settings['transfersteps'] * empty_micro))
            df_eps_transfer = np.zeros((settings['transferend'] - settings['validationend'], settings['transfersteps'] * empty_micro * 3))
            df_sig_transfer = np.zeros((settings['transferend'] - settings['validationend'], settings['transfersteps'] * empty_micro * 3))

    # Include stress data
    # Reshape to (N, t_data, 3); then select (N, t_used, 3)
    homstress = df_stress.to_numpy().astype(float).reshape(-1, settings['timesteps'], 3)[:,:settings['steps_ahead']]

    # Delete files to save memory
    del( df_load )
    del( df_stress )
    del( macro_vec_unnorm )
    if settings['validationend'] != settings['transferend']:
        del( macro_vec_unnorm_transfer )


    # Read data and obtain boundaries for normalization, obnoxious because every line different length
    print("Loading strains...")
    if trim_for_testing:
        df_eps = polars.read_csv(settings['datafolder'] + '/strains_trimmed_expanded.data', has_header=False, separator=',').with_columns(polars.all().cast(polars.Float32, strict=False))
        df_sig = polars.read_csv(settings['datafolder'] + '/stress_trimmed_expanded.data', has_header=False, separator=',').with_columns(polars.all().cast(polars.Float32, strict=False))
        df_epspeq = polars.read_csv(settings['datafolder'] + '/epspeq_trimmed_expanded.data', has_header=False, separator=',').with_columns(polars.all().cast(polars.Float32, strict=False))
    else:
        df_eps = polars.read_csv(settings['datafolder'] + '/strains_expanded.data', has_header=False, separator=',').with_columns(polars.all().cast(polars.Float32, strict=False))
        df_sig = polars.read_csv(settings['datafolder'] + '/stress_expanded.data', has_header=False, separator=',').with_columns(polars.all().cast(polars.Float32, strict=False))
        df_epspeq = polars.read_csv(settings['datafolder'] + '/epspeq_expanded.data', has_header=False, separator=',').with_columns(polars.all().cast(polars.Float32, strict=False))

    # Convert to numpy for more flexibility
    df_epspeq = df_epspeq.to_numpy()
    df_eps = df_eps.to_numpy()
    df_sig = df_sig.to_numpy()

    # Only find normalizing bounds based on training and validation sets
    if not norm_given:
        eps_mean = np.nanmean(df_eps[:settings['validationend']])
        eps_std = np.nanstd(df_eps[:settings['validationend']])
        sig_mean = np.nanmean(df_sig[:settings['validationend']])
        sig_std = np.nanstd(df_sig[:settings['validationend']])

        # Normalization based on the std of non-zero epsp and epspeq values
        epspeq_std = np.nanstd(df_epspeq[df_epspeq[:settings['validationend']].nonzero()])

    eps_set = []
    sig_set = []
    epspeq_set = []
    
    # The following could more easily be done in a single loop.
    # However, this part is the main memory bottleneck. To prevent out-of-memory issues, this is done individually.
    for i in range(settings['transferend']):
        if i < settings['validationend']:
            eps_set.append(df_eps[i, :][~np.isnan(df_eps[i])])
        else:       # Add transfer data to same datasets
            j = i - settings['validationend']
            eps_set.append(df_eps_transfer[j, :][~np.isnan(df_eps_transfer[j])])
    # Reduce memory cost by deleting the original data files
    del (df_eps)

    for i in range(settings['transferend']):
        if i < settings['validationend']:
            sig_set.append( df_sig[i, :][~np.isnan(df_sig[i])] )
        else:
            j = i - settings['validationend']
            sig_set.append(df_sig_transfer[j, :][~np.isnan(df_sig_transfer[j])])
    del (df_sig)

    if settings['epspeq_feature']:
        for i in range(settings['transferend']):
            if i < settings['validationend']:
                epspeq_set.append(df_epspeq[i, :][~np.isnan(df_epspeq[i])])
            else:
                j = i - settings['validationend']
                epspeq_set.append(df_epspeq_transfer[j, :][~np.isnan(df_epspeq_transfer[j])])
    del (df_epspeq)

    if settings['validationend'] != settings['transferend']:
        del (df_eps_transfer)
        del (df_sig_transfer)
        del (df_epspeq_transfer)

    print("Obtained bounds. Adding noise...")

    # Initialize Normalization
    if not norm_given:
        elemStrainNorm = normUtils.normUnitvar([], normmean=eps_mean, normstd = eps_std, donorm=settings['norm_targets'] )        # mean of means and mean of stds not perfectly sound
        elemStressNorm = normUtils.normUnitvar([], normmean=sig_mean, normstd = sig_std, donorm=settings['norm_targets'] )

    # (LEGACY) Add noise to the inputs, linearly scaled with the strainnorm
    eps_set_noise = []
    # ... (removed) ...

    print("Normalizing strains...")

    # sig_set_normalized = copy.deepcopy(sig_set)
    sig_set_normalized = sig_set

    # Apply normalization to full field
    for i in range(settings['transferend']):
        eps_set[i] = elemStrainNorm.normalize(eps_set[i])
        sig_set_normalized[i] = elemStressNorm.normalize(sig_set_normalized[i])
        if settings['noisy_inputs']:
            eps_set_noise[i] = elemStrainNorm.normalize(eps_set_noise[i])
        if sum(np.isnan(eps_set[i])) != 0:
            print(f"eps_set[{i}] Tensor contains NaN")
            exit()

    print("Normalized strains. Normalizing eps_p & eps_p_eq...")

    if not norm_given:  # Initialize normalization even when not used.
        # Not normalizing epsp, as not needed in data.
        elemepspNorm = normUtils.normUnitvar(normmean=0, normstd=1, donorm=False)

        # Normalization based on the std of non-zero epsp and epspeq values
        # No perfect way to do this. Many 0 values. Some high outliers. With Unitvar there will be some high values (>10). Alternative is everything between 0-1 instead.
        elemepspeqNorm = normUtils.normUnitvar(epspeq_set, normmean=0, normstd=epspeq_std, donorm=settings['norm_inputs'])

    if settings['epspeq_feature']:
        for i in range(settings['transferend']):
            epspeq_set[i] = elemepspeqNorm.normalize(epspeq_set[i])
            if sum(np.isnan(epspeq_set[i])) != 0:
                print(f"epspeq_set[{i}] Tensor contains NaN")
                exit()

    if not norm_given:
        homStressNorm = normUtils.normUnitvar(homstress, donorm=settings['norm_targets'], normmean=0)

    homstress_train = homStressNorm.normalize(homstress)
    if sum(np.isnan(homstress.reshape(-1))) != 0:
        print(f"Homogenized stress tensor contains NaN")
        exit()

    homstress_transfer = []
    if settings['validationend'] != settings['transferend']:
        if empty_micro == False:
            homstress_transfer = df_stress_transfer.to_numpy().astype(float).reshape(-1, settings['transfersteps'], 3)[:,:settings['transfersteps']]
        else:
            homstress_transfer = np.zeros((settings['transferend'] - 20, settings['transfersteps'], 3))
        homstress_transfer = homStressNorm.normalize(homstress_transfer)

    return fibers_set, macro_vec, macro_vec_transfer, eps_set, eps_set_noise, epspeq_set, macroStrainNorm, elemStrainNorm, elemepspNorm, elemepspeqNorm, homstress_train, homstress_transfer, homStressNorm, sig_set_normalized, elemStressNorm
