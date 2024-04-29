"""
Read the configuration file.
All possible variables need to be set, even when not used.
"""


import configparser
import json

def read_exp_config(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    return {
        # Graph setup
        'type' : config['setup']['type'],
        'name' : config['setup']['name'],
        'model' : config['setup']['model'],
        'trim': config['setup']['trim'] == 'True',
        'print_stiffness': config['setup']['print_stiffness'] == 'True',
        'plot_field': config['setup']['plot_field'] == 'True',
        'plot_finalonly': config['setup']['plot_finalonly'] == 'True',

        'plot_representative': json.loads(config['plotting']['plot_representative']),
        'plot_train': json.loads(config['plotting']['plot_train']),
        'plot_val': json.loads(config['plotting']['plot_val']),
        'plot_extra': json.loads(config['plotting']['plot_extra']),

        'extrapolate_num' : json.loads(config['specifics']['extrapolate_num']),
        'stiff_mode': config['specifics']['stiff_mode'],
        'representative_based_on': config['specifics']['representative_based_on'],
        'time_exrap_baseline': config['specifics']['time_exrap_baseline'] == 'True',
    }

def print_exp_config(fileloc, settings):
    keys = settings.keys()
    vals = settings.values()

    with open(f'{fileloc}', 'w') as file:
        for key, val in zip(keys, vals):

            # Exceptions
            except_keys = []
            if key in except_keys:
                continue

            # Headers, hard-coded
            if key == 'type':
                file.write('[setup]\n')
            if key == 'plot_train':
                file.write('\n[plotting]\n')
            if key == 'extrapolate_num':
                file.write('\n[specifics]\n')

            # Main content
            file.write(f'{key} = {val}')
            file.write('\n')

    return