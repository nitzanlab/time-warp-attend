import torch
import numpy as np
from .spline_flows import NSF_CL, NormalizingFlow


def augment_normalizing_flow(DE, augment_type='NSF_CL', nlayers=1, add_actnorm=False, dim=2, **kwargs_aug): 
    """
    Edited from https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib1.ipynb
    """
    coords = DE.coords_xy
    dim = DE.dim
    xy = coords.reshape(-1, 2)
    
    # compute fixed point coordinates
    flows = []
    fixed_pts_org = DE.get_fixed_pts()
    fixed_pts_org = torch.tensor(fixed_pts_org).type(torch.FloatTensor).reshape((-1,2)) if fixed_pts_org is not None else None
    fixed_pts = None

    # construct a model
    if augment_type == 'NSF_CL':
        nfs_flow = NSF_CL
        flows = [nfs_flow(dim=dim, **kwargs_aug) for _ in range(nlayers)]  
        
        # construct the model
        model = NormalizingFlow(flows)
        zs, _ = model.backward(xy)
        z1 = zs[-1]
        vectors_new = DE.forward(0, z=z1)

    else:

        raise NotImplementedError(f'augment_type={augment_type} not implemented')
    
    vectors_new = vectors_new.detach().numpy().reshape(coords.shape)
    # TEMP: print where (0,0) has been mapped to
    if fixed_pts_org is not None:
        fixed_pts = model.forward(fixed_pts_org)[0][-1]
        fixed_pts = fixed_pts.detach().numpy()
    return vectors_new, fixed_pts
