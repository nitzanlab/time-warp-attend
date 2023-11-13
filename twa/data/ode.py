import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import math
import torch
import numpy as np
from torchdiffeq import odeint, odeint_adjoint
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from twa.data.polynomials import sindy_library, library_size
from time import time
from sklearn import linear_model
from twa.utils.utils import ensure_device
from scipy.stats import binned_statistic_dd
from scipy.interpolate import interpn
from scipy.spatial.distance import cdist, pdist, squareform
from twa.data.topology import topo_dont_know, get_topology_Jacobian, topo_rep_spiral, topo_period_attr
from matplotlib import colorbar
from matplotlib.colors import Normalize


def get_value(input_val, class_val, default_val):
    """
    Returns most specific value that is not None
    """
    val = input_val
    if val is None:
        val = class_val
        if val is None:
            val = default_val
    return val

class FlowSystemODE(torch.nn.Module):
    """
    Super class for all systems of Ordinary Differential Equations.
    Default is a system with one parameter over two dimensions
    """
    
    # ODE params
    n_params = 1
    
    # recommended param ranges
    recommended_param_ranges = [[-1, 1]] * n_params

    # configs excluded from info
    exclude = ['params', 'solver', 'library', 'coords_ij', 'coords_xy']

    eq_string = None
    plot_param_idx = [0,1]

    labels = ['x', 'y']

    min_dims = [-1, -1]
    max_dims = [1, 1]

    foldx = 1
    foldy = 1

    shiftx = 0
    shifty = 0

    def __init__(self, params=None, labels=None, adjoint=False, solver_method='euler', train=False, device='cuda',
                 num_lattice=64, min_dims=None, max_dims=None, boundary_type=None, boundary_radius=1e1, boundary_gain=1.0,
                 foldx=None, foldy=None, shiftx=None, shifty=None, 
                 time_direction=1, **kwargs):
        """
        Initialize ODE. Defaults to a 2-dimensional system with a single parameter

        params - torch array allocated for system params (default: 0)
        labels - labels of the params (default: ['x', 'y'])
        adjoint - torch solver for differential equations (default: using odeint)
        solver_method - solver method (default: 'euler')
        train - boolean setting whether the parameters are trainable (default: False)
        device - (default: 'cuda')
        num_lattice - resolution of lattice of initial conditions
        min_dims - dimension's minimums of lattice of initial conditions
        max_dims - dimension's maximums of lattice of initial conditions
        boundary_type - type of boundary condition (default: None)
        """
        super().__init__()
        self.solver = odeint_adjoint if adjoint else odeint
        self.solver_method = solver_method
        n_params = len(params) if params is not None else 1
        self.device = ensure_device(device)
        params = torch.zeros(n_params) if params is None else params
        self.params = torch.nn.Parameter(params, requires_grad=True) if train else params
        self.params = torch.tensor(self.params) if not isinstance(self.params, torch.Tensor) else self.params
        self.params = self.params.float()
        self.labels = self.__class__.labels if labels is None else labels
        self.dim = len(self.labels)
        self.num_lattice = num_lattice
        self.min_dims = self.__class__.min_dims if min_dims is None else min_dims
        self.max_dims = self.__class__.max_dims if max_dims is None else max_dims
        
        # self.min_dims = get_value(min_dims, self.__class__.min_dims,  [-1, ] * self.dim)
        # self.max_dims = get_value(max_dims, self.__class__.max_dims,  [1, ] * self.dim)
        
        # self.min_dims = min_dims if min_dims is not None else [-1, ] * self.dim
        # self.max_dims = max_dims if max_dims is not None else [1, ] * self.dim
        self.boundary_type = boundary_type
        self.boundary_radius = torch.tensor(boundary_radius).float()
        self.boundary_gain = torch.tensor(boundary_gain).float()
        self.boundary_box = [min_dims, max_dims]
        self.time_direction = time_direction
        self.poly_order = 3 
        self.polynomial_terms = sindy_library(torch.ones((1, self.dim)), poly_order=self.poly_order, include_sine=False, include_exp=False)[1]
        self.coords_ij = None
        self.coords_xy = None
        if self.dim == 2:
            self.coords_xy = self.generate_mesh(min_dims=self.min_dims, max_dims=self.max_dims, num_lattice=self.num_lattice, indexing='xy')
            self.coords_ij = self.generate_mesh(min_dims=self.min_dims, max_dims=self.max_dims, num_lattice=self.num_lattice, indexing='ij')
        self.foldx = self.__class__.foldx if foldx is None else foldx
        self.foldy = self.__class__.foldy if foldy is None else foldy
        self.shiftx = self.__class__.shiftx if shiftx is None else shiftx
        self.shifty = self.__class__.shifty if shifty is None else shifty
        assert len(self.min_dims) == self.dim, 'min_dims must be of length dim'
        assert len(self.max_dims) == self.dim, 'max_dims must be of length dim'
        assert len(self.labels) == self.dim, 'labels must be of length dim'

    


    def run(self, T, alpha, init=None, clip=True, time_direction=None):
        """
        Run system

        T - length of run
        alpha - resolution
        init - initial startpoint

        returns:
        """
        init = torch.zeros(self.dim, device=self.device) if init is None else init
        time_direction = self.time_direction if time_direction is None else time_direction
        T = T * time_direction
        grid = torch.linspace(0, T, abs(int(T / alpha)), device=self.device)
        trajectory = self.solver(self, init, grid, rtol=1e-3, atol=1e-5, method=self.solver_method)
        if clip:
            trajectory = torch.cat([torch.clamp(trajectory[...,i].unsqueeze(-1), min=self.min_dims[i], max=self.max_dims[i]) for i in range(self.dim)], dim=-1)
        return trajectory


    def same_lattice_params(self, min_dims=None, max_dims=None, num_lattice=None):
        """
        Returns whether same params
        """
        same_min_dims = min_dims == self.min_dims
        same_max_dims = max_dims == self.max_dims
        same_num_lattice = num_lattice == self.num_lattice
        
        return same_min_dims and same_max_dims and same_num_lattice


    def get_lattice_params(self, min_dims=None, max_dims=None, num_lattice=None):
        """
        Substitute lattice parameters if provided
        """
        min_dims = self.min_dims if min_dims is None else min_dims
        max_dims = self.max_dims if max_dims is None else max_dims
        num_lattice = self.num_lattice if num_lattice is None else num_lattice

        return min_dims, max_dims, num_lattice



    def get_vector_field(self, which_dims=[0,1], min_dims=None, max_dims=None, num_lattice=None, 
                        slice_lattice=None, coords=None, vectors=None, return_slice_dict=False):
        """
        Returns a vector field of the system
        """
        if coords is None:
            coords = self.generate_mesh(min_dims=min_dims, max_dims=max_dims, num_lattice=num_lattice, indexing='xy').to(self.device)
        dim = coords.shape[-1]
        if vectors is None:
            vector_dims = [self.num_lattice] * dim + [dim]
            vectors = self.forward(0, coords).detach().cpu().numpy().reshape(vector_dims)
            
        coords = coords.detach().cpu().numpy() if isinstance(coords, torch.Tensor) else coords

        # validate slice_lattice
        if slice_lattice is None:
            slice_lattice = [self.num_lattice // 2] * (dim)

        slice_lattice = np.array(slice_lattice)

        if (len(slice_lattice) < dim) or np.any(slice_lattice > self.num_lattice) or np.any(slice_lattice < 0):
            raise ValueError('slice_lattice must be of length dim and within the range of num_lattice')

        # slice vector field
        idx = [slice(None)] * len(coords.shape)
        slice_dict = {}
        for i in range(dim):
            if i not in which_dims:
                idx[i] = slice_lattice[i]
                slice_dict[i] = coords[(slice_lattice[i],) * dim][..., i]
        idx = tuple(idx)
        
        coords = coords[idx][..., which_dims]
        vectors = vectors[idx][..., which_dims]

        if return_slice_dict:
            return coords, vectors, slice_dict
        else:
            return coords, vectors



    def generate_mesh(self, min_dims=None, max_dims=None, num_lattice=None, indexing='ij'):
        """
        Creates a lattice (tensor) over coordinate range
        """
        if self.same_lattice_params(min_dims, max_dims, num_lattice) and self.coords_ij is not None:
            return self.coords_ij
        else:
            min_dims, max_dims, num_lattice = self.get_lattice_params(min_dims, max_dims, num_lattice)
            coords = [torch.linspace(mn, mx, num_lattice) for (mn, mx) in zip(min_dims, max_dims)]
            mesh = torch.meshgrid(*coords, indexing=indexing)
            return torch.cat([ms[..., None] for ms in mesh], axis=-1)


    def get_sand_image(self, T=10, alpha=0.01, exclude_boundary=False, weight=False):
        """
        Returns a sand image
        """
        eps_boundary = (self.max_dims[0] - self.min_dims[0]) / self.num_lattice # one pixel boundary
        coords = self.coords_xy.to(self.device).reshape(-1,self.dim)
        all_flow = self.run(init=coords, T=T, alpha=alpha, clip=False).detach().cpu().numpy()
        coords = coords.detach().cpu().numpy()
        
        sand_image = torch.zeros(self.num_lattice, self.num_lattice)
        flow = all_flow[-1] # last time step
        
        # not sure yet why nan but lets find last point before
        if np.isnan(all_flow).sum() > 0:
            idx = np.where(np.isnan(all_flow))[0][0]
            idx = idx - 1
            if idx < 1:
                return sand_image
            flow = all_flow[idx]

        if exclude_boundary:
            in_bounds = (flow > np.array(self.min_dims) + eps_boundary) & (flow < np.array(self.max_dims) - eps_boundary)
            idx = np.where(in_bounds.sum(axis=-1) == self.dim)[0]
            flow = flow[idx]
            
        if flow.shape[0] != 0:
            eps_div = 1e-6
            
            coords = coords.reshape(-1, self.dim)
            w = 1 / (cdist(coords, flow) + eps_div)
            wflow = np.einsum('sz,z->s', w, np.ones(flow.shape[0]))
            sand_image = wflow.reshape(self.num_lattice, self.num_lattice)
        
        return sand_image

    
    ############################################################ Plotting ############################################################


    @staticmethod
    def plot_trajectory_2d_(coords, vectors, title='', ax=None, density=1.0, which_dims=[0,1], xlabel='x', ylabel='y', fontsize=15, cmap='gist_heat'):
        """
        Plot a single trajectory
        """
        coords = coords.detach().cpu().numpy() if isinstance(coords, torch.Tensor) else coords
        vectors = vectors.detach().cpu().numpy() if isinstance(vectors, torch.Tensor) else vectors
        xdim, ydim = which_dims
        X = coords[..., xdim]
        Y = coords[..., ydim]
        U = vectors[..., xdim]
        V = vectors[..., ydim]
        R = np.sqrt(U**2 + V**2)
        if ax is None:
            fig, ax = plt.subplots()
        stream = ax.streamplot(X, Y, U, V, density=density, color=R, cmap=cmap, integration_direction='forward')
        
        ax.set_xlim([X.min(), X.max()])
        ax.set_ylim([Y.min(), Y.max()])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=fontsize)
        return stream
    
    @staticmethod
    def plot_angle_image_(vectors=None, angle=None, add_colorbar=False, alpha=1.0, title='', ax=None):
        if angle is None:
            if vectors is None:
                raise ValueError('Either angle or vectors must be provided')
            else:
                angle = torch.tensor(np.arctan2(vectors[...,1], vectors[...,0])).type(torch.FloatTensor) # angle is bw -pi and pi
        
        # plot angle bw 0 and 2pi
        n_angle = angle % (2*np.pi)
        nticks = 3
        if ax is None:
            fig, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'})
        
        norm = Normalize(0, 2 * np.pi)
        im = ax.imshow(n_angle, cmap=plt.cm.twilight_shifted, norm=norm, alpha=alpha)
        
        ax.invert_yaxis()
        ax.set_title(title, fontsize=15)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nticks))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nticks))
        if add_colorbar:
            cax, kw = colorbar.make_axes_gridspec(ax)
            cb = plt.colorbar(im, cax=cax, **kw)
            cb.set_ticks(np.arange(0, 2.1*np.pi, np.pi/2.))
            cb.set_ticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$',
                            r'$\frac{3\pi}{2}$', r'$2\pi$'])
            cb.set_label(r'$\theta$', fontsize='x-large')
            cb.ax.tick_params(labelsize='x-large')
        return angle

    def plot_angle_image(self, **kwargs):
        """
        Plot angle image
        """
        _, vectors = self.get_vector_field()
        angle = self.plot_angle_image_(vectors, **kwargs)
        return angle

    def plot_trajectory(self, ax=None, which_dims=[0,1], slice_lattice=None, density=1.0, coords=None, vectors=None, fontsize=15, title=''):
        """
        Plot multiple trajectories
        """
        xdim, ydim = which_dims
        coords, vectors, slice_dict = self.get_vector_field(which_dims=which_dims, slice_lattice=slice_lattice, coords=coords, vectors=vectors, return_slice_dict=True)
        if ax is None:
            fig, ax = plt.subplots(1)
        if len(slice_dict) > 0:
            title += '\n'
            title += ', '.join([f'{self.labels[i]} = {slice_dict[i].item():.2f}' for i in range(self.dim) if i not in which_dims])
        xlabel=self.labels[xdim]
        ylabel=self.labels[ydim]
        self.plot_trajectory_2d_(coords=coords, vectors=vectors, title=title, ax=ax, density=density, which_dims=which_dims, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize)
        

    def get_trajectories(self, n_trajs, T, alpha, inits=None, clip=False, min_dims=None, max_dims=None):
        """
        Get multiple trajectories
        """
        min_dims, max_dims, _ = self.get_lattice_params(min_dims=min_dims, max_dims=max_dims)
        inits = torch.tensor(np.random.uniform(min_dims, max_dims, (n_trajs, self.dim))) if inits is None else torch.tensor(inits, device=self.device)
        trajectories = []
        for init in inits:
            trajectory = self.run(T, alpha=alpha, init=init, clip=clip).detach().cpu().numpy()
            trajectories.append(trajectory)

        return trajectories
    
        
    
    def get_vector_field_from_trajectories(self, n_trajs, T, alpha, which_dims=[0,1], num_lattice=None, min_dims=None, max_dims=None, slice_lattice=None, trajectories=None):
        """
        Generates a vector field from multiple trajectories
        """
        trajectories = self.get_trajectories(n_trajs=n_trajs, T=T, alpha=alpha, min_dims=min_dims, max_dims=max_dims) if trajectories is None else trajectories
        coords = self.generate_mesh(min_dims=min_dims, max_dims=max_dims, num_lattice=num_lattice).cpu().numpy()
        
        # trajectory coordinates
        traj_coords = []
        for trajectory in trajectories:
            traj_coords.append((trajectory[:-1] + trajectory[1:]) / 2)
        traj_coords = np.array(traj_coords).reshape(-1, self.dim)
        
        # trajectory vectors
        traj_vectors = []
        for trajectory in trajectories:
            traj_vectors.append(np.diff(trajectory, axis=0) / alpha)
        traj_vectors = np.array(traj_vectors).reshape(-1, self.dim)
        # v = np.diff(trajectories, axis=0).reshape(-1, self.dim) / alpha

        # masking inf values
        mask = np.isfinite(traj_vectors).all(axis=1)
        if ~np.all(mask):
            print('Masking %d inf / %d values' % ((~mask).sum(), len(mask)))
            traj_coords = traj_coords[mask]
            traj_vectors = traj_vectors[mask]

        # binning
        range_ = [[mn,mx] for mn,mx in zip(self.min_dims, self.max_dims)]
        vs = []
        for d in np.arange(self.dim):
            v_d = binned_statistic_dd(traj_coords, traj_vectors[:,d], statistic='mean', bins=self.num_lattice, range=range_)
            vs.append(v_d.statistic)
        vectors = np.stack(vs, axis=-1)

        # slice
        coords, vectors = self.get_vector_field(coords=coords, vectors=vectors, which_dims=which_dims, num_lattice=num_lattice, min_dims=min_dims, max_dims=max_dims, slice_lattice=slice_lattice)
        
        return coords, vectors


    def params_str(self, s=''):
        """
        Returns a string representation of the system's parameters
        """
        if self.eq_string:
            s = s + self.eq_string % tuple(self.params)
        else:
            s = (','.join(np.round(np.array(self.params), 3).astype(str))) if s == '' else s
        return s


    def plot_vector_field(self, which_dims=[0, 1], ax=None, min_dims=None, max_dims=None, num_lattice=None, 
                          title='', slice_lattice=None, coords=None, vectors=None):
        """
        Plot the vector field induced on a lattice
        :param which_dims: which dimensions to plot
        :param slice_lattice: array of slice_lattice in lattice in each dimension (for which_dims dimensions all values are plotted)
        """
        if ax is None:
            fig, ax = plt.subplots(1)
        
        xdim, ydim = which_dims

        coords, vectors, slice_dict = self.get_vector_field(which_dims=which_dims, min_dims=min_dims, max_dims=max_dims, num_lattice=num_lattice, 
                                                            slice_lattice=slice_lattice, 
                                                            coords=coords, vectors=vectors, return_slice_dict=True)
        ax.quiver(coords[..., xdim], coords[..., ydim], 
                  vectors[..., xdim], vectors[..., ydim],)

        ax.set_xlabel(self.labels[xdim])
        ax.set_ylabel(self.labels[ydim])
        slice_str = ''
        if len(slice_dict) > 0:
            slice_str = '\n'
            slice_str += ', '.join([f'{self.labels[i]}={slice_dict[i]}' for i in slice_dict])
        ax.set_title(self.params_str(title) + slice_str)

    ############################################################ Saved information ############################################################

    def get_info(self, exclude=exclude):
        """
        Return dictionary with the configuration of the system
        """
        data_info = self.__dict__
        data_info = {k: v for k, v in data_info.items() if not k.startswith('_')}

        for p in exclude:
            if p in data_info.keys():
                _ = data_info.pop(p)

        for k,v in data_info.items():
            if isinstance(v, torch.Tensor):
                data_info[k] = v.tolist()

        return data_info


    ############################################################ Polynomial representation ############################################################    

    @staticmethod
    def fit_polynomial_representation_(coords, vectors, poly_order, fit_with='lstsq', return_rt=False, desc='', **kwargs):
        coords = coords.numpy() if isinstance(coords, torch.Tensor) else coords
        vectors = vectors.numpy() if isinstance(vectors, torch.Tensor) else vectors
        dim = coords.shape[-1]
        num_lattice = coords.shape[0]
        library, library_terms = sindy_library(coords.reshape(num_lattice**dim, dim), poly_order=poly_order)

        dx_list = []
        total_time = 0
        for d in range(dim):
            zx = vectors[...,d].flatten()
        
            if fit_with == 'lstsq':
                start = time()
                mx = np.linalg.lstsq(library, zx, rcond=None, **kwargs)[0]
                stop = time()
            elif fit_with == 'lasso':
                clf = linear_model.Lasso(**kwargs)
                start = time()
                clf.fit(library, zx)
                mx = clf.coef_
                stop = time()
            else:
                raise ValueError('fit_with must be either "lstsq" or "lasso"')
            dx = pd.DataFrame(data={lb:m.astype(np.float64) for lb,m in  zip(library_terms,mx)}, index=[desc])
            total_time += stop - start
            dx_list.append(dx)

        if return_rt:
            return *dx_list, total_time
        else:
            return dx_list
    
    def fit_polynomial_representation(self, coords=None, vectors=None, poly_order=None, **kwargs):
        """
        Return the polynomial representations of the system
        Assumes two dimensions, x and y
        :param poly_order: order of the polynomial
        :param fit_with: method to fit the polynomial, options are 'lstsq' or 'lasso'
        :param kwargs: additional arguments to pass to the fitting method
        """
        poly_order = poly_order if poly_order else self.poly_order
        coords, vectors = self.get_vector_field(coords=coords, vectors=vectors)
        return self.fit_polynomial_representation_(coords=coords, vectors=vectors, poly_order=poly_order, desc=self.__class__.__name__,  **kwargs)


    def get_polynomial_representation(self):
        """
        Return the polynomial representations of the system
        Assumes two dimensions, x and y
        """
        return None


    def get_library(self, z):
        if self.dim != 2:
            raise NotImplementedError('Polynomial system is only implemented for 2D systems.')
        if self.poly_order != 3:
            raise NotImplementedError('Polynomial system is only implemented for poly_order=3.')
        
        z = z.reshape(-1, self.dim)
        x = z[...,0]
        y = z[...,1]
        library = torch.stack([x**0, x, y, x**2, x*y, y**2, x**3,x**2*y, x*y**2, y**3], axis=-1).to(self.device).float()
        return library
    

    ############################################################ Invariances ############################################################
    @staticmethod
    def get_bifurcation_curve():
        print('Not implemented yet')
        return []

    def get_fixed_pts_org(self):
        """
        Returns the fixed points defined by the original system
        """
        print('Fixed pts original not implemented yet')
        return None

    def get_fixed_pts(self):
        """
        Returns fixed points coordinates after shift and scaling
        """
        pts_org = self.get_fixed_pts_org()
        if pts_org is None:
            return None
        pts = []
        for (x_st,y_st) in pts_org:
            x_st = x_st / self.foldx + self.shiftx
            y_st = y_st / self.foldy + self.shifty
            pts.append([x_st, y_st])
        return pts
    
    def J(self, x, y):
        """
        Returns the Jacobian of the system at (x,y)
        """
        print('Not implemented yet')
        return None
    
    def curl(self, x, y):
        """
        Returns the curl of the system at (x,y)
        """
        if self.dim != 2:
            raise NotImplementedError('Curl is only implemented for 2D systems.')
        
        A = self.J(x, y)
        if A is None:
            return None
        
        return A[1,0]-A[0,1]

    def get_topology_supercriticalhopf(self):
        """
        Given that the system undergoes a supercritical Hopf bifurcation and implementation of fixed points,
        return either cycle & unstable fixed point or stable fixed point
        """
        # fixed pt
        fixed_pts = self.get_fixed_pts()
        if fixed_pts is None:
            return None
        x_st, y_st = fixed_pts[0]
        J_st = self.J(x_st, y_st)
    
        topo = get_topology_Jacobian(J_st)
        topos = [topo]
        if topo == topo_rep_spiral:
            topos.append(topo_period_attr)
        return topos
    
    def get_topology(self):
        """
        Returns a list of topologies that the system has
        """
        return [topo_dont_know]


    def get_dist_from_bifur(self):
        """
        Returns the distance from the bifurcation point
        """
        pass
    
    @staticmethod
    def get_pts_isin_(pts, min_dims, max_dims, eps=1e-3):
        """
        """
        dim = len(min_dims)
        pts = pts.reshape(-1, dim)
        n = pts.shape[0]
        pts_isin = np.full(n, True)
        for i in range(dim):
            coord_isin = ((pts[:,i] < (max_dims[i] - eps)) & (pts[:,i] > (min_dims[i] + eps)))
            pts_isin = pts_isin & coord_isin 
        
        return pts_isin
        
    def get_pts_isin(self, pts, **kwargs):
        """
        Returns a boolean array indicating whether the points are in the domain
        """
        return self.get_pts_isin_(pts, min_dims=self.min_dims, max_dims=self.max_dims, **kwargs)

