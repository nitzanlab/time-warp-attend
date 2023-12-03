import numpy as np
import pandas as pd

topo_dont_know = -1

topo_none = 0 

# point classes:
topo_attractor = 1
topo_saddle = 2
topo_repeller = 3
topo_attr_spiral = 4
topo_rep_spiral = 5
topo_degenerate = 6
topo_center = 7
topo_line = 8

# periodic orbits:
topo_period_attr = 9 
topo_period_rep = 10 
topo_period_in_attr_out_rep = 11 
topo_period_in_rep_out_attr = 12

topo_ghost = 13

topo_num_to_str_dict = {
    topo_dont_know: 'dont know',
    topo_ghost: 'ghost',
    topo_attractor: 'Point', #'attractor point',
    topo_saddle: 'saddle point',
    topo_repeller: 'repeller point',
    topo_attr_spiral:  'Point', #'spiral to attractor point',
    topo_rep_spiral:  'Periodic', #'spiral from repeller point',
    topo_degenerate: 'degenerate',
    topo_center: 'centers',
    topo_line: 'lines',
    topo_period_attr: 'Periodic', #'attracting periodic orbit', 
    topo_period_rep: 'repelling periodic orbit',
    topo_period_in_attr_out_rep: 'attracting inside repelling outside periodic orbit',
    topo_period_in_rep_out_attr: 'attracting outside repelling inside periodic orbit',
}

topo_num_to_point_vs_cycle_desc = {
    topo_dont_know: 'dont know',
    topo_ghost: 'dont know',
    topo_attractor: 'point',
    topo_saddle: 'dont know',
    topo_repeller: 'dont know',
    topo_attr_spiral: 'point',
    topo_rep_spiral: 'cycle',
    topo_degenerate: 'dont know',
    topo_center: 'dont know',
    topo_line: 'dont know',
    topo_period_attr: 'cycle',
    topo_period_rep: 'dont know',
    topo_period_in_attr_out_rep: 'dont know',
    topo_period_in_rep_out_attr: 'dont know',
}


def topo_to_str(topo_num):
    """
    Return a string representation of the topology.
    """
    return '\n'.join([f' ({itopo+1}) {topo_num_to_str_dict[topo]}' for itopo, topo in enumerate(topo_num)])

pt_attr_idx = 0 # index of Point label
cyc_attr_idx = 1 # index of Cycle label

def topo_point_vs_cycle(topo=None):
    """
    Return True if the topology is a cycle, False if it is a point.
    [1,0] - Point
    [0,1] - Cycle
    
    """
    topo_values = {'Point': [topo_attractor, topo_attr_spiral],
                   'Periodic': [topo_period_attr]}
    nsamples = 0 if topo is None else topo.shape[0]
    label = pd.DataFrame(np.zeros((nsamples, topo_values.keys().__len__())), columns=topo_values.keys())

    if nsamples == 0:
        return label
    
    topo[topo < 0] = 0
    # topo = topo if topo.shape == 2 else topo.reshape(-1,2)
    topo = topo.astype(int)
    
    for i, (k, vs) in enumerate(topo_values.items()):
        label[k] = np.stack([(topo==v) for v in vs]).sum(axis=0).sum(axis=-1).astype(int)

    return label
    

def get_topology_Jacobian(A):
    """
    Given the Jacobian matrix A, return the topology of the fixed point.
    """
    tr = np.trace(A)
    det = np.linalg.det(A)
    delta = tr**2 - 4*det

    if det < 0:
        # Saddle
        topo = topo_saddle #0
    elif det > 0:
            if tr > 0:
                if det < delta:
                    # Source
                    topo = topo_repeller #1
                elif det > delta:
                    # Spiral source
                    topo = topo_rep_spiral #2
                else:
                    # Degenerate source
                    topo = topo_degenerate #-1
            elif tr < 0:
                if det < delta:
                    # Sink
                    topo = topo_attractor #3
                elif det > delta:
                    # Spiral sink
                    topo = topo_attr_spiral #4
                else:
                    # Degenerate sink
                    topo = topo_degenerate # -1
            else:
                # Center
                topo = topo_center #-1
    else:
        # Line
        topo = topo_line # -1
    return topo



from scipy.optimize import minimize

def get_dist_from_bifur_curve(a, b, f, a_limits=None):
    """
    Computes distance of a,b points to the curve defined by (a,f(a)). Sign corresponds to the side of the curve.
    ab_s: (n,2) array of points corresponding to a,b parameters
    f: function of a
    """
    def obj(x):
        y = f(x)
        return np.sqrt((x - a)**2 + (y - b)**2)
    res = minimize(obj, x0=a, bounds=[a_limits])
    a_opt = res.x
    dist = obj(a_opt)
    return dist
