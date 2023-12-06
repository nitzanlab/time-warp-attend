import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def get_optimizer(optimizer_name, model_params, lr, momentum=0.0):
    """
    Select optimizer
    optimizer_name - name of optimizer (Adam/SGD)
    model_params - model weights/parameters
    lr - learning rate
    """
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model_params, lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model_params, lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model_params, lr=lr, momentum=momentum)
    else:
        raise ValueError('Optimizer "{}" not recognized.'.format(optimizer_name))
    return optimizer

def permute_data(X):
    '''Permute data to (batch, dim, ...grid...)'''
    grid_dims = len(X.shape) - 2 
    permutation = tuple([0,-1] + [i+1 for i in range(grid_dims)]) # permute to (batch, dim, ...grid...)
    X = X.permute(permutation)
    return X
    
def rev_permute_data(X):
    '''Reverse permute data from (batch, dim, ...grid...) to (batch, ...grid..., dim)'''
    grid_dims = len(X.shape) - 2
    permutation = tuple([0] + [i+2 for i in range(grid_dims)] + [1]) # permute to (batch,...grid...,dim)
    X = X.permute(permutation)
    return X
