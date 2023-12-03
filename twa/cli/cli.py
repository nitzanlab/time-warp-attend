import os
import click
import time
import numpy as np
import pandas as pd
import time

from twa.train import predict_model, VecTopoDataset, train_model_alt
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
@click.option('--add-traj', '-traj', is_flag=True, default=False)
@click.option('--augment-type', '-at', type=click.Choice([None, 'NSF_CL']), default=None)
@click.option('--augment-ntries', type=int, default=10, help="Number of times to try to augment each sample")
@click.option('--NSF-B', type=float, default=4, help="Range being warped by NSF")
@click.option('--NSF-K', type=int, default=5, help="Number of spline knots in NSF")
@click.option('--NSF-hd', type=int, default=6, help="Number of hidden layers in NSF")

@click.option('--device', type=str, default='cpu')
@click.option('--seed', '-se', type=int, default=0)
@click.option('--config-file', type=click.Path())
def generate_dataset(data_dir, data_name, train_size, test_size, sampler_type, num_lattice, min_dims, max_dims, labels, 
param_ranges, noise_type, noise_mag, add_sand, add_traj, augment_type, augment_ntries, nsf_b, nsf_k, nsf_hd, device, seed, config_file):
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
    
    ensure_dir(data_dir)

    start = time.time()
    param_ranges = param_ranges if param_ranges is None else [str_to_list(x) for x in param_ranges] #TODO: better handle!
    sf = SystemFamily(data_name=data_name, num_lattice=num_lattice, min_dims=min_dims, max_dims=max_dims, labels=labels, param_ranges=param_ranges, device=device, seed=seed)
    
    num_samples = train_size + test_size
    res = sf.generate_flows(num_samples=num_samples, 
                            noise_type=noise_type, 
                            noise_level=noise_mag,
                            sampler_type=sampler_type, 
                            add_sand=add_sand,
                            add_traj=add_traj,
                            augment_type=augment_type, 
                            augment_ntries=augment_ntries,
                            B=nsf_b, K=nsf_k, hidden_dim=nsf_hd,)

    params_pert = res['params_pert']
    vectors_pert = res['vectors_pert']
    DEs_pert = res['DEs_pert']
    poly_params_pert = res['poly_params_pert']
    sand_pert = res['sand_pert']
    trajs_pert = res['trajs_pert']
    fixed_pts_pert = res['fixed_pts_pert']
    dists_pert = res['dists_pert']
    topos_pert = res['topos_pert']

    savenames = ['X', 'p', 'sysp', 'fixed_pts', 'dists', 'topo']
    split = [vectors_pert, poly_params_pert, params_pert, fixed_pts_pert, dists_pert, topos_pert]
    if add_traj:
        savenames += ['trajs']
        split += [trajs_pert]

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
    sf_info['with_traj'] = add_traj
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
@click.option('--train-data-descs', '-t', type=str, multiple=True, default=['simple_oscillator_nsfcl'])
@click.option('--outdir', '-o', type=click.Path(), default='output/')
@click.option('--data-dir', '-dd', type=click.Path())
@click.option('--datatype', '-dt', type=click.Choice(['angle', 'vector', 'param']), default='angle')
@click.option('--model-type', '-mt', type=str, default='AttentionwFC')
@click.option('--no-attention', is_flag=True, default=False)
@click.option('--repeats', type=int, default=50)
@click.option('--dont-save', is_flag=True, default=False)
@click.option('--verbose', is_flag=True, default=False)
@click.option('--dropout-rate', type=float, default=0.9)
@click.option('--kernel-size', type=int, default=3)
@click.option('--latent-dim', type=int, default=10)
@click.option('--batch-size', type=int, default=64)
@click.option('--conv-layers', type=int, default=4)
@click.option('--conv-dim', type=int, default=64)
@click.option('--datasize', type=int, default=10000)
@click.option('--lr', type=float, default=1e-4)
@click.option('--num-epochs', type=int, default=20)
@click.option('--desc', type=str, default='')
@click.option('--device', type=str, default='cuda')
@click.option('--test-data-descs', type=str, multiple=True, default=[
                                                        'simple_oscillator_noaug',
                                                        'simple_oscillator_nsfcl',
                                                        'lienard_poly',
                                                        'lienard_sigmoid',
                                                        'vanderpol',
                                                        'suphopf',
                                                        'subhopf',
                                                        'bzreaction',
                                                        'selkov2',
                                                        'pancreas_clusters_random_bin',
                                                        'repressilator',
                                                        ])
@click.option('--test-noise', is_flag=True, default=False)
@click.option('--seed', type=int, default=0)
@click.option('--pretrained-path', type=str)
@click.option('--config-file', type=click.Path())
def call_train(train_data_descs, outdir, data_dir, datatype, model_type, no_attention, test_data_descs, test_noise, repeats, dont_save, verbose, dropout_rate, 
               kernel_size, latent_dim, batch_size, conv_layers, conv_dim, datasize, lr, num_epochs, desc, seed, device, pretrained_path, config_file):
    """
    Train vector field topology classifier.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.isdir(outdir):
        raise ValueError(f'Outdir {outdir} does not exist')
    data_dir = os.path.join(outdir, 'data') if data_dir is None else data_dir
    if not os.path.isdir(data_dir):
        raise ValueError(f'Data dir {data_dir} does not exist')
    
    device = ensure_device(device)
    
    with_attention = not no_attention
    save = not dont_save

    with_attention_str = 'atten' if with_attention else 'noatten'
    
    train_data_desc = '_'.join(train_data_descs)
    if model_type == 'AttentionwFC':
        exp_name = f'{train_data_desc}_{datatype}_{with_attention_str}_{model_type}' if desc == '' else desc
    else:
        exp_name = f'{train_data_desc}_{datatype}_{model_type}' if desc == '' else desc


    print(exp_name)

    
    results_dir = os.path.join(outdir, 'results')
    ensure_dir(results_dir)
    
    train_info = {'train_data_desc': train_data_desc,
                    'datatype': datatype,
                    'model_type': model_type,
                    'with_attention': with_attention,
                    'repeats': repeats,
                    'dropout_rate': dropout_rate,
                    'kernel_size': kernel_size,
                    'latent_dim': latent_dim,
                    'batch_size': batch_size,
                    'conv_layers': conv_layers, 
                    'conv_dim': conv_dim,
                    'datasize': datasize,
                    'lr': lr,
                    'num_epochs': num_epochs,
                    'desc': desc,
                    'device': device,
                    'pretrained_path': pretrained_path,
                    'seed': seed}
    

    # test datasets
    tt = 'test'
    test_datasets = {}
    for test_data_desc in test_data_descs:
        print(test_data_desc)
        test_data_dir = os.path.join(data_dir, test_data_desc)
        if os.path.isdir(test_data_dir):
            try:
                test_dataset = VecTopoDataset(test_data_dir, tt=tt, datatype=datatype)
            except:
                print(f'Could not load {test_data_dir}')
                continue
            test_datasets[test_data_desc] = test_dataset

            try:
                test_dataset = VecTopoDataset(test_data_dir, tt=tt, datatype=datatype, noise=0.1)
            except:
                print(f'Could not load {test_data_dir}')
                continue
            test_datasets[test_data_desc + '_noise0.1'] = test_dataset


            try:
                test_dataset = VecTopoDataset(test_data_dir, tt=tt, datatype=datatype, noise=0.2)
            except:
                print(f'Could not load {test_data_dir}')
                continue
            test_datasets[test_data_desc + '_noise0.2'] = test_dataset


            try:
                test_dataset = VecTopoDataset(test_data_dir, tt=tt, datatype=datatype, noise=0.3)
            except:
                print(f'Could not load {test_data_dir}')
                continue
            test_datasets[test_data_desc + '_noise0.3'] = test_dataset


        else:
            print(f'{test_data_dir} does not exist')
        
    # optional: adding noise/mask testing
    if test_noise:
        noises = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # mask_probs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3,0.35, 0.4, 0.45, 0.5]
        test_data_desc_base = train_data_desc
        test_data_dir = os.path.join(data_dir, test_data_desc_base)
        
        for noise in noises:
            test_data_desc = test_data_desc_base + '_noise%.2f' % noise
            test_dataset = VecTopoDataset(test_data_dir,  tt=tt, datatype=datatype, noise=noise) 
            test_datasets[test_data_desc] = test_dataset
        
        # for mask_prob in mask_probs:
        #     test_data_desc = test_data_desc_base + '_masked%.2f' % mask_prob
        #     test_dataset = VecTopoDataset(test_data_dir,  tt=tt, datatype=datatype, mask_prob=mask_prob) 
        #     test_datasets[test_data_desc] = test_dataset


    for i in range(repeats):
        print(f'Run {i}')
        exp_desc = exp_name + '_' + str(i)

        # read train data
        for itrain_data_desc, train_data_desc in enumerate(train_data_descs):
            train_data_dir = os.path.join(data_dir, train_data_desc)
            if itrain_data_desc == 0:
                train_dataset = VecTopoDataset(train_data_dir, datatype=datatype, datasize=datasize, filter_outbound=True)
                train_dataset.plot_data()
            else:
                train_dataset += VecTopoDataset(train_data_dir, datatype=datatype, datasize=datasize, filter_outbound=True)

        model, _ = train_model_alt(train_dataset, model_type=model_type, with_attention=with_attention, lr=lr, kernel_size=kernel_size, num_epochs=num_epochs,
                      dropout_rate=dropout_rate, batch_size=batch_size, conv_layers=conv_layers, conv_dim=conv_dim, device=device, verbose=verbose, latent_dim=latent_dim, pretrained_path=pretrained_path)
        

        # evaluate on test data
        res = []
        save_dir = None
        for test_data_desc, test_dataset in test_datasets.items():
            exp_results_dir = os.path.join(results_dir, exp_desc)
    
            if save:
                ensure_dir(exp_results_dir)
                train_config = os.path.join(exp_results_dir, 'train_config.yaml')
                write_yaml(train_config, train_info)
            
                save_dir = os.path.join(exp_results_dir, test_data_desc)
                ensure_dir(save_dir)

            correct, test_loss, _ = predict_model(model, test_dataset, device=device, verbose=False, save=save, save_dir=save_dir, is_prob=(dropout_rate > 0))
            res.append({'data': test_data_desc,
                        'correct': correct,
                        'loss': test_loss})

        print(pd.DataFrame(res))

    print('Successfully trained {}. Config file: {}'.format(exp_desc, exp_results_dir))