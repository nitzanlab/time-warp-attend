import os
import sys
import click
import uuid
import time
import numpy as np
import pandas as pd
import time
from twa.cli.utils import PythonLiteralOption

from twa.train import train_model, predict_model, VecTopoDataset
from .utils import command_with_config
from twa.utils import ensure_dir, ensure_device, write_yaml, str_to_list
from twa.data import SystemFamily
from sklearn.model_selection import train_test_split

import torch
import random

# Living dangerously
import warnings
warnings.filterwarnings("ignore")
# Here we will monkey-patch click Option __init__
# in order to force showing default values for all options
orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init  # type: ignore


@click.group()
@click.version_option()
def cli():
    pass

################################################## Single Experiment ###################################################

@cli.command(name="generate-dataset", cls=command_with_config('config_file'), context_settings=dict(ignore_unknown_options=True, allow_extra_args=True), 
help="Generates a data set of vector fields.")
@click.option('--data-dir', '-f', type=str)
@click.option('--data-name', '-d', type=str, default='dataset')
@click.option('--train-size', '-m', type=int, default=0)
@click.option('--test-size', '-m', type=int, default=0)
@click.option('--sampler-type', '-sp', type=str, default='random')
@click.option('--num-lattice', '-n', type=int, default=64) # 
@click.option('--min-dims', '-mi', type=list)
@click.option('--max-dims', '-mn', type=list)
@click.option('--labels', '-mx', type=list)
@click.option('--param-ranges', '-pr', type=list)
@click.option('--noise-type', '-nt', type=click.Choice([None, 'gaussian', 'masking', 'parameter', 'trajectory']), default=None)
@click.option('--noise-mag', '-n', type=float, default=0.0)
@click.option('--add-sand', '-sand', is_flag=True, default=False)

@click.option('--augment-type', '-at', type=click.Choice([None, 'NSF_CL']), default=None)
@click.option('--augment-ntries', type=int, default=10, help="Number of times to try to augment each sample")
@click.option('--NSF-B', type=float, default=4, help="Range being warped by NSF")
@click.option('--NSF-K', type=int, default=5, help="Number of spline knots in NSF")
@click.option('--NSF-hd', type=int, default=6, help="Number of hidden layers in NSF")

@click.option('--device', type=str, default='cpu')
@click.option('--seed', '-se', type=int, default=0)
@click.option('--config-file', type=click.Path())
def generate_dataset(data_dir, data_name, train_size, test_size, sampler_type, num_lattice, min_dims, max_dims, labels, 
param_ranges, noise_type, noise_mag, add_sand, augment_type, augment_ntries, nsf_b, nsf_k, nsf_hd, device, seed, config_file):
    """
    Generates train and test data for one data set

    Positional arguments:

    data_name (str): name of data set and folder to save all data in.
    system_names (list of str): names of data to generate
    num_samples (int): number of total samples to generate
    sampler_type (list of strings): for each system, a string denoting the type of sampler_type used. 
    system_props (list of floats): for each system, a float controlling proportion of total data this system will comprise.
    val_size (float): proportion in (0,1) of data allocated to validation set
    num_lattice (int): number of points for all dimensions in the equally spaced grid on which velocity is measured
    min_dims (list of floats): the lower bounds for each dimension in phase space
    max_dims (list of floats): the upper bounds for each dimension in phase space
    moise_type (None, 'gaussian', 'masking', 'parameter'): type of noise to apply to the data, including None. Gaussian means white noise on the vector field; masking means randomly zeroing out vectors; parameter means gaussian noise added to the parameters
    noise_mag (float): amount of noise, interpreted differently according to each noise type. If gaussian, then the std of the applied noise relative to each vector field's natural std; if masking, proportio to be masked; if parameter, then just the std of applied noise. 
    seed (int): random seed
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # save_dir = os.path.join(data_dir, data_name)
    ensure_dir(data_dir)

    # read config file
    # data_info = {}
    # if config_file is not None:
    #     data_info = read_yaml(config_file)
    # param_ranges = strtuple_to_list(param_ranges)
    start = time.time()
    param_ranges = param_ranges if param_ranges is None else [str_to_list(x) for x in param_ranges] #TODO: better handle!
    sf = SystemFamily(data_name=data_name, num_lattice=num_lattice, min_dims=min_dims, max_dims=max_dims, labels=labels, param_ranges=param_ranges, device=device, seed=seed)
    
    num_samples = train_size + test_size
    res = sf.generate_flows(num_samples=num_samples, 
                            noise_type=noise_type, 
                            noise_level=noise_mag,
                            sampler_type=sampler_type, 
                            add_sand=add_sand,
                            augment_type=augment_type, 
                            augment_ntries=augment_ntries,
                            B=nsf_b, K=nsf_k, hidden_dim=nsf_hd,)

    params_pert = res['params_pert']
    vectors_pert = res['vectors_pert']
    DEs_pert = res['DEs_pert']
    poly_params_pert = res['poly_params_pert']
    sand_pert = res['sand_pert']
    fixed_pts_pert = res['fixed_pts_pert']
    dists_pert = res['dists_pert']
    topos_pert = res['topos_pert']

    savenames = ['X', 'p', 'sysp', 'fixed_pts', 'dists', 'topo']
    split = [vectors_pert, poly_params_pert, params_pert, fixed_pts_pert, dists_pert, topos_pert]

    if add_sand:
        savenames += ['sand']
        split += [sand_pert]
    
    if (test_size > 0) and (train_size > 0):
        split = train_test_split(*split, test_size=test_size, train_size=train_size, random_state=seed)
    
    tt = ['train'] * (train_size > 0) + ['test'] * (test_size > 0)
    filenames = [f'{s}_{t}' for s in savenames for t in tt]

    for dt, nm in zip(split, filenames):
        np.save(os.path.join(data_dir, nm + '.npy'), dt)

    # save config file
    data_config = os.path.join(data_dir, 'data_config.yaml')
    sf_info = sf.get_sf_info()
    sf_info['data_dir'] = data_dir
    sf_info['with_sand'] = add_sand
    sf_info['noise_type'] = noise_type
    if noise_type is not None:
        sf_info['noise_level'] = noise_mag
    sf_info['sampler_type'] = sampler_type
    sf_info['augment_type'] = augment_type
    if augment_type is not None:
        sf_info['augment_ntries'] = augment_ntries
        sf_info['nsf_b'] = nsf_b
        sf_info['nsf_k'] = nsf_k
        sf_info['nsf_hd'] = nsf_hd

    write_yaml(data_config, sf_info)

    print('Successfully generated data for {}. Config file: {}'.format(data_name, data_config))
    end = time.time()
    print('Time elapsed: {}'.format(end - start))


@cli.command(name='train', cls=command_with_config('config_file'), help='Train vector field topology classifier.')
@click.argument("train-data-desc", type=click.Path())
@click.argument("outdir", type=click.Path())
@click.option('--vectors', is_flag=True, default=False)
@click.option('--no-attention', is_flag=True, default=False)
@click.option('--repeats', type=int, default=50)
@click.option('--dont-save', is_flag=True, default=False)
@click.option('--verbose', is_flag=True, default=False)
@click.option('--dropout-rate', type=float, default=0.9)
@click.option('--kernel-size', type=int, default=3)
@click.option('--latent-dim', type=int, default=10)
@click.option('--batch-size', type=int, default=64)
@click.option('--conv-dim', type=int, default=64)
@click.option('--datasize', type=int, default=10000)
@click.option('--lr', type=float, default=1e-4)
@click.option('--num-epochs', type=int, default=20)
@click.option('--desc', type=str, default='')
@click.option('--device', type=str, default='cuda')
@click.option('--test-data-descs', type=list, default=['simple_oscillator_noaug',
                                                        'simple_oscillator_nsfcl',
                                                        'suphopf',
                                                        'bzreaction',
                                                        'selkov',
                                                        'lienard_poly',
                                                        'lienard_sigmoid',
                                                        'pancreas_clusters_random_bin'])
@click.option('--seed', type=int, default=0)
@click.option('--config-file', type=click.Path())
def call_train(train_data_desc, outdir, vectors, no_attention, test_data_descs, repeats, dont_save, verbose, dropout_rate, kernel_size, latent_dim, batch_size, 
               conv_dim, datasize, lr, num_epochs, desc, seed, device, config_file):
    """
    Train vector field topology classifier.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.isdir(outdir):
        raise ValueError(f'Outdir {outdir} does not exist')
    data_dir = os.path.join(outdir, 'data')
    if not os.path.isdir(data_dir):
        raise ValueError(f'Data dir {data_dir} does not exist')
    
    device = ensure_device(device)
    
    to_angle = not vectors
    with_attention = not no_attention
    save = not dont_save

    to_angle_str = 'angle' if to_angle else 'vector'
    with_attention_str = 'atten' if with_attention else 'noatten'
    
    exp_name = f'{train_data_desc}_{to_angle_str}_{with_attention_str}' if desc == '' else desc

    print(exp_name)


    # test datasets
    
    results_dir = os.path.join(outdir, 'results')
    
    ensure_dir(results_dir)
    
    tt = 'test'

    test_datasets = {}
    for test_data_desc in test_data_descs:
        test_data_dir = os.path.join(data_dir, test_data_desc)
        if os.path.isdir(test_data_dir):
            test_dataset = VecTopoDataset(test_data_dir, tt=tt, to_angle=to_angle) 
            test_datasets[test_data_desc] = test_dataset
        else:
            print(f'{test_data_dir} does not exist')
        

    for i in range(repeats):
        print(f'Run {i}')
        exp_desc = exp_name + '_' + str(i)

        # train
        train_data_dir = os.path.join(data_dir, train_data_desc)
        train_dataset = VecTopoDataset(train_data_dir, to_angle=to_angle, datasize=datasize, filter_outbound=True)
            
        model, _ = train_model(train_dataset, to_angle=to_angle, with_attention=with_attention, lr=lr, kernel_size=kernel_size, num_epochs=num_epochs,
                      dropout_rate=dropout_rate, batch_size=batch_size, conv_dim=conv_dim, device=device, verbose=verbose, latent_dim=latent_dim)
        
        # evaluate on test data
        exp_results_dir = os.path.join(results_dir, exp_desc)
        ensure_dir(exp_results_dir)


        res = []
        save_dir = None
        for test_data_desc, test_dataset in test_datasets.items():
            
            if save:
                save_dir = os.path.join(exp_results_dir, test_data_desc)
                ensure_dir(save_dir)

            correct, test_loss, _ = predict_model(model, test_dataset, device=device, verbose=False, save=save, save_dir=save_dir, is_prob=(dropout_rate > 0))
            res.append({'data': test_data_desc,
                        'correct': correct,
                        'loss': test_loss})

        print(pd.DataFrame(res))
    