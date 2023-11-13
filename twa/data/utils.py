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




def interp_vectors(cell_coords, cell_vectors, dim=2, num_lattice=64):
    """
    Interpolate vector field on a lattice
    """
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

        interp = interpolate.Rbf(x_src, y_src, f_src, function='linear', smooth=2) #0.1)
        z = interp(grid_x, grid_y)
        
        
        vectors.append(z) 

    vectors = np.stack(vectors, axis=-1)
    coords = np.stack([grid_x, grid_y], axis=-1)
    return coords, vectors
