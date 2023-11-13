import os
import torch
import pickle
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .ode_classes import *
from twa.utils.utils import ensure_device, ensure_dir
from sklearn.model_selection import train_test_split


import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch import distributions
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from torch.nn.parameter import Parameter

from .augmentations import augment_normalizing_flow
from .ode import FlowSystemODE





class SystemFamily():
    """
    Family of ODE or PDE systems
    """

    @staticmethod
    def get_generator(data_name):
        """
        Selecting supported ODE or PDE generator
        """
        if data_name == 'simple_oscillator' or data_name == 'so':
            generator = SimpleOscillator

        # elif data_name == 'point_attractor' or data_name == 'pa':
        #     generator = PointAttractor

        elif data_name == 'selkov':
            generator = Selkov

        elif data_name == 'suphopf':
            generator = SupercriticalHopf

        elif data_name == 'prey_preditor' or data_name == 'pp':
            generator = PreyPredator

        elif data_name == 'bzreaction' or data_name == 'bz':
            generator = BZreaction

        elif data_name == 'fitzhugh_nagumo' or data_name == 'fn':
            generator = FitzhughNagumo

        elif data_name == 'vanderpol' or data_name == 'vp':
            generator = VanDerPol

        elif data_name == 'biased_vanderpol' or data_name == 'bvp':
            generator = BiasedVanDerPol

        elif data_name == 'polynomial':
            generator = Polynomial

        elif data_name == 'lienard_poly':
            generator = LienardPoly
        
        elif data_name == 'lienard_sigmoid':
            generator = LienardSigmoid

        elif data_name == 'infinite_period' or data_name == 'ip':
            generator = InfinitePeriod

        elif data_name == 'repressilator':
            generator = Repressilator
        else:
            raise ValueError(f'Unknown data, `{data_name}`! Try `simple_oscillator`.')

        return generator


    @staticmethod
    def get_sampler(sampler_type):
        """
        Selecting supported sampler
        """
        if (sampler_type == 'uniform') or (sampler_type == 'random'):
            sampler = SystemFamily.params_random
        elif sampler_type == 'extreme':
            sampler = SystemFamily.params_extreme
        elif sampler_type == 'sparse':
            sampler = SystemFamily.params_sparse
        elif sampler_type == 'control':
            sampler = SystemFamily.params_control
        elif sampler_type == 'constant':
            sampler = SystemFamily.params_constant
        else:
            print(sampler_type)
            raise ValueError('Param sampler not recognized.')

        return sampler



    def __init__(self, data_name, device=None, #min_dims=None, max_dims=None, num_lattice=None, labels=None, 
                param_ranges=None, param_groups=None, seed=0, **kwargs):
        """
        Generate a system family
        :param data_name: name of system
        :param param_ranges: range for each param in model, 
        :param device: device to use
        :param min_dims: minimum range of dimensions to use
        :param max_dims: maximum range of dimensions to use
        :param kwargs: any arguments of a general system
        """

        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)

        self.data_name = data_name
        self.pde = False
         
        DE = SystemFamily.get_generator(self.data_name)
        # self.data_dir = os.path.abspath(data_dir) if data_dir is not None else '.'

        # if not provided, use ode suggested params
        self.param_ranges = DE.recommended_param_ranges if param_ranges is None else param_ranges
        self.param_groups = DE.recommended_param_groups if param_groups is None else param_groups
        # self.labels = DE.labels if labels is None else labels
        # self.min_dims = DE.min_dims if min_dims is None else min_dims
        # self.max_dims = DE.max_dims if max_dims is None else max_dims
        param_use_ode_defaults = ['min_dims', 'max_dims', 'labels', 'num_lattice']
        for p in param_use_ode_defaults:
            if p in kwargs.keys() and kwargs[p] is None:
                kwargs.pop(p)
        # general DE params
        self.device = ensure_device(device)
        params = self.params_random(1)
        DE_ex = DE(params=params[0], device=device, **kwargs)
        
        
        data_info = DE_ex.get_info()
        # data_info = {**self.__dict__, **data_info} # merge dictionaries
        self.data_info = data_info
        # self.num_lattice = num_lattice
        self.dim = DE_ex.dim
        self.DE = DE
        self.DE_ex = DE_ex
        if self.data_name != 'grayscott': # TODO: ask isinstance(generator, ODE/PDE)
            # TODO: can generator.param can be used as default?
            self.coords = self.DE_ex.generate_mesh() # min_dims, max_dims, num_lattice
            self.coords = self.coords.to(device).float()

        self.kwargs = kwargs

        # self.param_sampler = SystemFamily.get_sampler(sampler_type)

    def get_sf_info(self):
        sf_info = self.data_info.copy()
        sf_info['param_ranges'] = self.param_ranges
        sf_info['data_name'] = self.data_name
        return sf_info

######################################## Param sampling methods ########################################################

    def params_random(self, num_samples):
        """
        Return random sampling of params in range
        """
        params = np.zeros((num_samples, len(self.param_ranges)))
        
        for i,p in enumerate(self.param_ranges):
            params[:, i] = np.random.uniform(low=p[0], high=p[1], size=num_samples)
        return params
    
    def params_constant(self, num_samples, fill_value=None):
        """
        Return constant params
        """
        param_values = [np.quantile([-1,1], 0.6) for p in self.param_ranges] if fill_value is None else fill_value
        params = np.repeat([param_values], num_samples, axis=0)
        
        return params

    def params_extreme(self, num_samples=None):
        """
        Return array of extreme (minimal, and maximal bounds) param combinations
        """
        # param_bounds = [torch.tensor([mn, mx]) for (mn, mx) in self.param_ranges]
        # mesh = torch.meshgrid(*param_bounds)
        # params = torch.cat([ms[..., None] for ms in mesh], dim=-1)

        nparams = len(self.param_ranges)
        lst = list(itertools.product([0, 1], repeat=nparams))
        params = np.zeros((len(lst), len(self.param_ranges)))
        for i, p in enumerate(lst):
            params[i] = [self.param_ranges[j][p[j]] for j in range(nparams)]

        return params

    def params_sparse(self, num_samples, p=0.5):
        """
        Samples which parameters to set with Binomial distribution with probability p and then 
        randomly assigns them (for families of many parameters)
        """
        which_params = np.random.binomial(1, p, (num_samples, len(self.param_ranges)))
        params = self.params_random(num_samples)
        return (params * which_params)

    def params_control_single(self, max_coeff=3., prop_zero=.6, prop_non_unit=.3):

        # proportion of unit parameters
        prop_unit = 1 - prop_zero - prop_non_unit

        # Initialize parameter vector
        num_terms = int(len(self.param_ranges) / self.dim)
        x =  (2*max_coeff) * torch.rand(num_terms, self.dim)

        # zero out `prop_zero` parameters
        coeffs = torch.where(x/(2*max_coeff) < prop_zero, torch.zeros_like(x), x)

        # Add 1 coeffs
        coeffs = torch.where((x/(2*max_coeff) >= prop_zero)*(x/(2*max_coeff)< (prop_zero + prop_unit/2)), torch.ones_like(coeffs), coeffs)

        # Add -1 coeffs
        coeffs = torch.where((x/(2*max_coeff) >=prop_zero + prop_unit/2)*(x/(2*max_coeff) < (prop_zero + prop_unit)), -1*torch.ones_like(coeffs), coeffs)

        # Add random coeffs
        coeffs = torch.where(x/(2*max_coeff)>prop_zero + prop_unit, (2*max_coeff) * torch.rand(num_terms, self.dim) - max_coeff, coeffs)

        # Are both equations identically 0?
        one_zero_eq = (coeffs.sum(0)[0] * coeffs.sum(0)[1] == 0)
        if one_zero_eq:
            # Make some parameters randomly +/- 1
            for i in range(self.dim):
                ind = np.random.randint(num_terms)
                sgn  = 2 * torch.rand(1) - 1
                sgn /= sgn.abs()
                coeffs[ind,i] = sgn * 1.
        return coeffs.reshape(-1).numpy() #.unsqueeze(0).numpy()

    def params_control(self, num_samples, **kwargs): #TODO: TEMP
        """
        Return array of control parameters (all parameters are zero except one)
        """
        params = np.zeros((num_samples, len(self.param_ranges)))
        for i in range(num_samples):
            params[i] = self.params_control_single(**kwargs)
        return params
    
######################################## Flow noise functions ########################################################

    def noise_vectors_gaussian(self, vectors, noise_level):
        """
        Add gaussian noise to vectors
        """
        noise = np.random.normal(scale=noise_level, size=vectors.shape)
        return vectors + noise

    def noise_vectors_mask(self, vectors, noise_level, empty_val=np.nan):
        """
        Add mask noise to vectors
        """
        vectors = vectors.copy()
        mask = np.random.uniform(size=np.array(vectors.shape)[:-1]) < noise_level
        vectors[mask, :] = empty_val #
        return vectors

    def noise_vectors(self, vectors, noise_type, noise_level):
        """
        Add noise to parameters
        """
        if noise_type == 'gaussian':
            return self.noise_vectors_gaussian(vectors, noise_level)
        if noise_type == 'mask':
            return self.noise_vectors_mask(vectors, noise_level)
        else:
            return vectors

    


    def augment_normalizing_flow_with_rejection(self, DE, ntries=10, **kwargs_aug):
        """
        Augment normalizing flow with rejection sampling
        :param DE: FlowSystemODE
        """
        
        for i in range(ntries):
            
            vectors_new, fixed_pts = augment_normalizing_flow(DE, **kwargs_aug)
            if fixed_pts is None:
                break

            fixed_pts_isin = DE.get_pts_isin(fixed_pts,)
            # is_inbound = True
            # for fp in fixed_pts:
            #     inbound = [ (fp[i] >= DE.min_dims[i]) & (fp[i] <= DE.max_dims[i]) for i in range(DE.dim)]
            #     is_inbound = is_inbound & np.all(inbound)
            
            
            if np.all(fixed_pts_isin): # checking if all points are inside
                break
            
        
        if (i == ntries-1):
            print(fixed_pts)
            print(i)
            print('Could not find a flow that flows inside the coords boundaries')


        return vectors_new, fixed_pts









######################################## Params noise functions ########################################################

    
    def noise_params(self, params, noise_type, noise_level):
        """
        Add noise to parameters
        """
        if noise_type == 'params_gaussian':
            return self.noise_vectors_gaussian(params, noise_level)
        else:
            return params

######################################## Data generation ########################################################

    def generate_flows(self, num_samples, noise_type=None, noise_level=None, sampler_type='random', params=None, 
                      interpolate_missing=False, add_sand=False, augment_type=None, augment_ntries=10, max_topos=3, **kwargs_aug):
        """
        Generate original and perturbed params and flows
        """
        
        sampler = SystemFamily.get_sampler(sampler_type)
        params = params if params else sampler(self, num_samples)
        
        params_pert = self.noise_params(params, noise_type, noise_level=noise_level)
        
        vectors = []
        vectors_pert = []

        DEs = []
        DEs_pert = []

        sand = []
        sand_pert = []

        fixed_pts = []
        fixed_pts_pert = []

        dists = []
        dists_pert = []

        topos = []
        topos_pert = []

        poly_params = []
        poly_params_pert = []


        for p, p_pert in zip(params, params_pert):

            DE = self.DE(params=p, **self.data_info)
            coords, v = DE.get_vector_field()
            
            DE_pert = self.DE(params=p_pert, **self.data_info)

            if noise_type == 'trajectory':
                print('Works well in odes but not here, check!') # TODO: fix this
                _, v_pert = DE_pert.get_vector_field_from_trajectories(n_trajs=noise_level, T=5, alpha=0.01)
                # plt.quiver(coords[...,0], coords[...,1], v[...,0], v[...,1])
                # plt.quiver(coords[...,0], coords[...,1], v_pert[...,0], v_pert[...,1], color='r')
            else:
                _, v_pert = DE_pert.get_vector_field()

            # fixed points
            fp = DE.get_fixed_pts()
            if augment_type:
                v_pert, fp_pert = self.augment_normalizing_flow_with_rejection(DE_pert, augment_type=augment_type, ntries=augment_ntries, **kwargs_aug)
            else:
                fp_pert = DE_pert.get_fixed_pts()
            
            # distance from bifurcation
            dist = DE.get_dist_from_bifur()
            dist_pert = DE_pert.get_dist_from_bifur()
            
            no_pert = False
            if np.all(np.isclose(v, v_pert)):
                no_pert = True

            if add_sand:
                image = DE.get_sand_image()
                image_pert = image if no_pert else DE_pert.get_sand_image()
                sand.append(image)
                sand_pert.append(image_pert)

            
            tps = DE.get_topology()
            if len(tps) < max_topos:
                tps += [0] * (max_topos - len(tps))
            if len(tps) > max_topos:
                print(f'WARNING: more than {max_topos} topologies found for {self.data_name}. Truncating to first {max_topos}.')
                tps = tps[:max_topos]
            
            pp = DE.get_polynomial_representation()
            pp = pp if pp else DE.fit_polynomial_representation()
            if np.isclose(v, v_pert).all(): # no perturbation
                pp_pert = pp
            else:
                pp_pert = DE_pert.fit_polynomial_representation(coords=coords, vectors=v_pert)
            
            pp = np.stack(pp).T.flatten()
            pp_pert = np.stack(pp_pert).T.flatten()
            
            
            fixed_pts_pert.append(fp_pert)
            fixed_pts.append(fp)
            dists_pert.append(dist_pert)
            dists.append(dist)
            vectors.append(v)
            vectors_pert.append(v_pert)
            DEs.append(DE)
            DEs_pert.append(DE_pert)
            topos.append(tps)
            topos_pert.append(tps)
            poly_params.append(pp)
            poly_params_pert.append(pp_pert)

        

        vectors = np.stack(vectors)
        vectors_pert  = np.stack(vectors_pert)

        sand = np.stack(sand) if len(sand) else None
        sand_pert  = np.stack(sand_pert) if len(sand_pert) else None

        vectors_pert = self.noise_vectors(vectors_pert, noise_type, noise_level=noise_level)

        topos = np.stack(topos)
        topos_pert  = np.stack(topos_pert)
        
        dists = np.stack(dists)
        dists_pert  = np.stack(dists_pert)
        
        poly_params = np.stack(poly_params)
        poly_params_pert  = np.stack(poly_params_pert)

        return {'params_pert': params_pert.astype('float32'), 
               'vectors_pert': vectors_pert, 
               'DEs_pert': DEs_pert, 
               'poly_params_pert': poly_params_pert, 
               'sand_pert': sand_pert, 
               'fixed_pts_pert': fixed_pts_pert, 
               'dists_pert': dists_pert, 
               'topos_pert': topos_pert, 
               'params': params.astype('float32'), 
               'vectors': vectors, 
               'DEs': DEs, 
               'poly_params': poly_params, 
               'sand': sand, 
               'fixed_pts': fixed_pts, 
               'dists': dists,
               'topos': topos
                }

    
######################################## Plotting ########################################################

    def plot_noised_vector_fields(self, num_samples, noise_type, noise_level, params=None, add_trajectories=False, title='', **kwargs):
        """
        Plot original and perturbed vector fields
        """
        _, flow_pert, DEs_pert, _, _, flow, DEs, _ = self.generate_flows(num_samples=num_samples, params=params, noise_type=noise_type, noise_level=noise_level, **kwargs)
        # self.plot_vector_fields(params_pert, flow_pert, params, flow, **kwargs)
        nrows = num_samples
        ncols = 2 + (add_trajectories * 2)
        fig, axs = plt.subplots(nrows, ncols, figsize=(10,5), tight_layout=False, constrained_layout=True)
        for i, (f, f_pert, DE, DE_pert) in enumerate(zip(flow, flow_pert, DEs, DEs_pert)):
            DE.plot_vector_field(vectors=f, ax=axs[i, 0])
            DE_pert.plot_vector_field(vectors=f_pert, ax=axs[i, 1 + add_trajectories])
            if add_trajectories:
                DE.plot_trajectory(vectors=f, ax=axs[i, 1])
                DE_pert.plot_trajectory(vectors=f_pert, ax=axs[i, 3])
        plt.suptitle(title)
        plt.show()
    
    def plot_vector_fields(self, params=None, sampler_type='uniform', add_trajectories=False, **kwargs):
        """
        Plot vector fields of system
        :param params: array of params for system
        :param param_selection: plot extreme (minimal, intermediate and maximal bounds) param combinations
        :param kwargs: additional params for sampling method
        """
        sampler = SystemFamily.get_sampler(sampler_type)
        params = sampler(self, **kwargs)
        num_samples = params.shape[0]
        skip = 1

        nrow = ncol = int(np.ceil(np.sqrt(num_samples)))
        if add_trajectories:
            ncol = 2
            nrow = num_samples
            skip = 2
        
        fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 6*nrow), tight_layout=False, constrained_layout=True)
        axs = axs.flatten()
        for i in range(num_samples):
            ax = axs[skip*i]
            model = self.generate_model(params[i, :])
            model.plot_vector_field(ax=ax)
            if add_trajectories:
                ax = axs[skip*i+1]
                model.plot_trajectory(ax=ax)
    
        plt.tight_layout()
        plt.show()

