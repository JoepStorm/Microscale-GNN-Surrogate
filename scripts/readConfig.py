"""
Read the configuration file.

To train multiple models, a single config file can be given, for which settings can be overwritten in "get_batch_settings" based on an additional parameter.
E.g.: "python trainGNN.py 3" will pass 3 as the setting_state.
"""

import configparser
import json

import numpy as np

def read_config(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    return {
        # Graph setup
        'fibers_considered' : json.loads(config['Graph setup']['fibers_considered']),
        'num_fib_rve' : json.loads(config['Graph setup']['num_fib_rve']),
        'vfrac' : json.loads(config['Graph setup']['vfrac']),
        'r_fiber' : json.loads(config['Graph setup']['r_fiber']),
        'self_loops' : config['Graph setup']['self_loops'] == 'True',
        'edge_weight_scale' : json.loads(config['Graph setup']['edge_weight_scale']),
        'periodicity' : config['Graph setup']['periodicity']  == 'True',
        'edge_augmentation' : config['Graph setup']['edge_augmentation'],  #None, DistFeats, NoFeats
        'augmentation_percentage': json.loads(config['Graph setup']['augmentation_percentage']),
        'edge_vec' : config['Graph setup']['edge_vec']  == 'True',
        # Training
        'trim_for_test' : config['Training']['trim_for_test'] == 'True', # This is not default, but a logic test!
        'fullfield_stress' : config['Training']['fullfield_stress'] == 'True',
        'stress_as_input_field' : config['Training']['stress_as_input_field'] == 'True',
        'track_loss_per_timestep' : config['Training']['track_loss_per_timestep'] == 'True',
        'layers_arr' : json.loads(config['Training']['layers_arr']),
        'hidden_nodes' : json.loads(config['Training']['hidden_nodes']),
        'activation_func' : config['Training']['activation_func'],
        'min_epochs' : json.loads(config['Training']['min_epochs'])+1,  # Perhaps not theoretically sound, but more intuitive
        'max_epochs' : json.loads(config['Training']['max_epochs'])+1,
        'max_train_hours': json.loads(config['Training']['max_train_hours']),
        'early_stop_epochs' : json.loads(config['Training']['early_stop_epochs']),
        'update_epochs' : json.loads(config['Training']['update_epochs']),
        'val_frequency' : json.loads(config['Training']['val_frequency']),
        'full_val_frequency' : json.loads(config['Training']['full_val_frequency']),
        'transf_frequency' : json.loads(config['Training']['transf_frequency']),
        'aggregation' : config['Training']['aggregation'],
        'seed' : json.loads(config['Training']['seed']),
        'batch_size' : json.loads(config['Training']['batch_size']),
        'bias' : config['Training']['bias'] == 'True',
        'normlayers' : config['Training']['normlayers'],
        'dropout' : json.loads(config['Training']['dropout']),
        'res_lay' : config['Training']['res_lay'] == 'True',
        'tot_res_lay' : config['Training']['tot_res_lay'] == 'True',
        'unique_MPLs' : config['Training']['unique_MPLs'] == 'True',
        'SGD_optim' : config['Training']['SGD_optim'] == 'True',
        'adaptive_lr' : config['Training']['adaptive_lr'] == 'True',
        'learn_rate' : json.loads(config['Training']['learn_rate']),
        'weight_decay': json.loads(config['Training']['weight_decay']),
        'main_error_train' : config['Training']['main_error_train'] == 'True',
        'pretrain': config['Training']['pretrain'],
        'xi_strainfield': json.loads(config['Training']['xi_strainfield']),
        'xi_stressfield': json.loads(config['Training']['xi_stressfield']),
        'xi_stresshom': json.loads(config['Training']['xi_stresshom']),
        'steps_ahead' : json.loads(config['Training']['steps_ahead']),
        # Data
        'timesteps' : json.loads(config['Data']['timesteps']),
        'transfersteps' : json.loads(config['Data']['transfersteps']),
        'transferdir' : config['Data']['transferdir'],
        'trainsize': json.loads(config['Data']['trainsize']),
        'random_trainset' : config['Data']['random_trainset'] == 'True',
        'trainend' : json.loads(config['Data']['trainend']),
        'validationend' : json.loads(config['Data']['validationend']),
        'transferend' : json.loads(config['Data']['transferend']),
        'transfertrain': config['Data']['transfertrain'] == 'True',
        'storetransfhom': config['Data']['storetransfhom'] == 'True',
        'plot_cases' : json.loads(config['Data']['plot_cases']),
        'plot_timesteps' : json.loads(config['Data']['plot_timesteps']),
        'load_vec_in_data' : config['Data']['load_vec_in_data'] == 'True',
        'load_vec_separate' : config['Data']['load_vec_separate'] == 'True',
        'load_vec_as_feature' : config['Data']['load_vec_as_feature'] == 'True',
        'epspeq_feature' : config['Data']['epspeq_feature'] == 'True',
        'numvoid_feature' : config['Data']['numvoid_feature'] == 'True',
        'norm_inputs' : config['Data']['norm_inputs'] == 'True',
        'norm_targets' : config['Data']['norm_targets'] == 'True',
        'noisy_inputs' : config['Data']['noisy_inputs'] == 'True',
        'savemodel' : config['Data']['savemodel'] == 'True'
    }


def get_batch_settings(settings, setting_state):
    """
    This function is used to set the hyperparameters based on the setting_state.
    This overwrites the settings set in the config file (fem/configGNN).
    """

    settings['seed'] = setting_state
    datasetsize = settings['validationend']

    # Set with a variable number of voids by default
    settings['meshdir'] = f'../meshes/fib1-{settings["num_fib_rve"]}_mesh{datasetsize}_t{settings["timesteps"]}'
    settings['datafolder'] = f'../data/fib1-{settings["num_fib_rve"]}_mesh{datasetsize}_t{settings["timesteps"]}'
    settings['saveresultdir'] = f'../results/fib1-{settings["num_fib_rve"]}_mesh{datasetsize}_t{settings["steps_ahead"]}'

    settings['transferdata'] = f"../data/{settings['transferdir']}"
    settings['transfermesh'] = f"../meshes/{settings['transferdir']}"

    # setup for "gp" -> Gaussian-process based load paths.
    settings['meshdir'] += '_gp'
    settings['datafolder'] += '_gp'
    settings['saveresultdir'] += '_gp'

    return settings


def printConfig(fileloc, settings):
    keys = settings.keys()
    vals = settings.values()

    with open(f'{fileloc}', 'w') as file:
        for key, val in zip(keys, vals):

            # Exceptions
            except_keys = ['transferdata', 'transfermesh', 'meshdir', 'datafolder', 'saveresultdir']
            if key in except_keys:
                continue

            # Headers, hard-coded
            if key == 'fibers_considered':
                file.write('[Graph setup]\n')
            if key == 'trim_for_test':
                file.write('\n[Training]\n')
            if key == 'timesteps':
                file.write('\n[Data]\n')

            # Main content
            file.write(f'{key} = {val}')
            file.write('\n')

    return
