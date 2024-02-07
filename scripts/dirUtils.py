"""
(Dirty) utilities file to create directories when training many models at the same time.
A temporary folder name is created during training.
After training, the foldername is changed based on a number of hyperparameters and model results to visually see how a model was trained by just the folder name.
"""

import os

def make_dir(base_folder, batch_job):
    if not batch_job:
        for i in range(1000):
            tmp_dirname = f"{base_folder}/tmp_{i}"
            if os.path.exists(tmp_dirname):
                continue
            else:
                os.mkdir(tmp_dirname)
                break
        print(f"Using temporary directory {tmp_dirname}")
    else:
        tmp_dirname = f"{base_folder}/tmp_{batch_job}"
        if os.path.exists(tmp_dirname):     # super dirty calling itself to get into loop above, but makes adapting stuff myself much more easy while being robust on server
            tmp_dirname = make_dir(base_folder, False)
        else:
            os.mkdir(tmp_dirname)
            print(f"Using temporary directory {tmp_dirname}")

    return tmp_dirname

def save_dir(cur_name, unnorm_hom_val_loss, train_loss, val_loss, transf_loss, settings, layers, fibers, epochs, time, use_cuda, batch_job):
    # Create dir to save results
    dirstring = f"{settings['saveresultdir']}/"
    dirstring += f"{unnorm_hom_val_loss:.5f}vhu_"
    if settings['main_error_train']:
        dirstring += f"{train_loss:.4f}train_{val_loss:.5f}val"
    else:
        dirstring += f"{val_loss:.5f}val_{train_loss:.4f}train"
    if settings['transferend'] != settings['validationend'] and settings['transfertrain']:
        dirstring += f"_{transf_loss:.4f}trnsf"
    dirstring += f"_{fibers}void_{settings['steps_ahead']}step_"
    dirstring += f"{layers}L{settings['hidden_nodes']}_{settings['aggregation']}_{settings['activation_func']}_{settings['normlayers']}Nrm_drop{settings['dropout']}_{epochs}E_"

    dirstring += "RMSE"
    if settings['noisy_inputs']:
        dirstring += '_noiseIn'
    dirstring += f"_{settings['seed']}_{time:.1f}s_tr{settings['trainsize']}_val{settings['validationend'] - settings['trainend']}_b{settings['batch_size']}"
    dirstring += f"_lr{settings['learn_rate']}"
    if settings['adaptive_lr']:
        dirstring += 'ad'
    dirstring += f"_xi_{settings['xi_strainfield']}_{settings['xi_stressfield']}_{settings['xi_stresshom']}"
    if not settings['epspeq_feature']:
        dirstring += f"_noEpspeq"
    if settings['fullfield_stress']:
        dirstring += f"_fullStress"
    else:
        dirstring += f"_fullStrain"
    if settings['stress_as_input_field']:
        dirstring += f"_StressTrain"
    dirstring += f"_{settings['edge_augmentation']}"
    if settings['edge_augmentation'] != 'None':
        dirstring += f"{settings['augmentation_percentage']}"

    if use_cuda:
        dirstring += '_GPU'
    else:
        dirstring += '_CPU'

    if batch_job != False:
        if batch_job < 10:
            dirstring += f'_job0{batch_job}'
        else:
            dirstring += f'_job{batch_job}'
    os.rename(cur_name, dirstring)
    print(f"Saved result to: {dirstring}")

    return
