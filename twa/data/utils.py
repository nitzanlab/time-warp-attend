import torch
import numpy as np
import os
from scipy import interpolate

def load_dataset(data_dir, savenames=['X', 'y', 'p'], tt=['train', 'test']):
    """
    Load dataset from data_dir
    """
    data = []
    filenames = [os.path.join(data_dir, f'{s}_{t}.npy') for s in savenames for t in tt]

    for filename in filenames:
        if os.path.exists(filename):
            data.append(np.load(filename))
    
    return data

def jacobian(f, spacings=1):
    """Returns the Jacobian of a batch of planar vector fields shaped batch x dim x spatial x spatial"""
    num_dims = f.shape[1]
    return torch.stack([torch.stack(torch.gradient(f[:,i], dim=list(range(1,num_dims+1)), spacing=spacings)) for i in range(num_dims)]).movedim(2,0)

def curl(f, spacings=1):
    """Returns the curl of a batch of 2d (3d) vector fields shaped batch x dim x spatial x spatial (x spatial)"""
    num_dims = f.shape[1]
    if num_dims > 4:
        raise ValueError('Curl is only defined for dim <=3.')
    elif num_dims < 3:
        b = f.shape[0]
        s = f.shape[-1]
        f = torch.tile(f.unsqueeze(-1),(s,))
        f = torch.cat((f, torch.zeros(b,1,s,s,s)), dim=1)
        spacings = [sp for sp in spacings]
        spacings.append(spacings[-1])
        spacings = tuple(spacings)

    J = jacobian(f,spacings=spacings)

    #[[dFxdz,dFxdy,dFxdx],[dFydz,dFydy,dFydx],[dFzdz,dFzdy,dFzdx]]

    dFxdy = J[:,0,1]
    dFxdz = J[:,0,2]
    dFydx = J[:,1,0]
    dFydz = J[:,1,2]
    dFzdx = J[:,2,0]
    dFzdy = J[:,2,1]
   
    return torch.stack([dFzdy - dFydz, dFxdz - dFzdx, dFydx - dFxdy]).movedim(1,0)

def divergence(f, spacings=1):
    """Returns the divergence of a batch of planar vector fields shaped batch x dim x spatial x spatial"""

    # J.shape = batch x dim x dim x [spatial]^n
    J = jacobian(f, spacings=spacings)
    return J[:,0,0] + J[:,1,1]

def laplacian(f):
    """Calculate laplacian of vector field"""
    num_dims = f.shape[1]
    if num_dims>3:
        raise ValueError('Laplacian not yet implemented for dim>2.')
    return torch.stack([divergence(torch.stack(torch.gradient(f[:,i], dim=[1,2])).movedim(1,0)) for i in range(num_dims)]).movedim(1,0)

# TODO: create evalution file?
def vector_field_lp_dist(vectors1, vectors2, p=2, coords1=None, coords2=None):
    """Calculate the lp distance between two vector fields
    :param vectors1: vector field 1 - shape (spatial x dim, dim) #(batch, dim, spatial, spatial)
    """
    if coords1 != coords2:
        # TODO: implement interpolation?
        raise ValueError('Vector fields must be defined on the same grid.')
    
    a = (vectors2 - vectors1) ** p
    a = np.ma.array(a, mask=np.isnan(a))
    a = np.power(np.sum(a, axis=-1), 1/p) # error per position
    err = np.mean(a)
    return err






def interp_vectors(cell_coords, cell_vectors, dim=2, num_lattice=64):

    min_dims = list(np.min(cell_coords, axis=0))
    max_dims = list(np.max(cell_coords, axis=0))

    x_dst = np.linspace(min_dims[0], max_dims[0], num_lattice)
    y_dst = np.linspace(min_dims[1], max_dims[1], num_lattice)
    x_src = cell_coords[:,0]
    y_src = cell_coords[:,1]

    vectors = []
    grid_x, grid_y = np.meshgrid(x_dst, y_dst)
    for d in range(dim): 
        f_src = cell_vectors[...,d]

        # interp = scipy.interpolate.NearestNDInterpolator(
        #     x = np.hstack([x_src.reshape(-1, 1), y_src.reshape(-1, 1)]),
        #     y = f_src.reshape(-1, ),
        # )
        # z = interp((grid_x, grid_y))
        
        interp = interpolate.Rbf(x_src, y_src, f_src, function='linear', smooth=2) #0.1)
        z = interp(grid_x, grid_y)
        
        
        vectors.append(z) 

    vectors = np.stack(vectors, axis=-1)
    coords = np.stack([grid_x, grid_y], axis=-1)
    return coords, vectors
