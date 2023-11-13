from .ode import FlowSystemODE
import pandas as pd
import torch
import os
import numpy as np
import torch.nn.functional as F
from twa.data.polynomials import sindy_library
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from .topology import *

eps = 1e-6

########################################################## helper methods ######################################################################

def cartesian_to_polar(x, y):
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return r, theta

def polar_derivative_to_cartesian_derivative(r, theta, rdot, thetadot):
    xdot = torch.cos(theta) * rdot - r * torch.sin(theta) * thetadot
    ydot = torch.sin(theta) * rdot + r * torch.cos(theta) * thetadot
    return xdot, ydot

########################################################## classical systems with hopf bifurcation ######################################################################

class SimpleOscillator(FlowSystemODE):
    """
    Simple oscillator:

        rdot = r * (a - r^2)
        xdot = rdot * cos(theta) - r * sin(theta)
        ydot = rdot * sin(theta) + r * cos(theta)

    Where:
        - a - angular velocity
        - r - the radius of (x,y)
        - theta - the angle of (x,y)
    """

    min_dims = [-1.,-1.]
    max_dims = [1.,1.]
    
    recommended_param_ranges=[[-0.5,0.5], [-1, 1]]
    recommended_param_groups=[recommended_param_ranges]
                    
    eq_string = r'$\dot{r} = r(%.02f - r^2); \dot{\theta} = %.02f$'
    short_name='so'
    
    bifurp_desc = r'$r$'
    param_descs = [r'$a$', r'$\omega$']
    

    def forward(self, t, z, **kwargs):
        x = z[..., 0]
        y = z[..., 1]

        a = self.params[0]
        omega = self.params[1]

        r,theta = cartesian_to_polar(x, y)
        
        rdot = r * (a - r ** 2)
        thetadot = omega

        xdot, ydot = polar_derivative_to_cartesian_derivative(r, theta, rdot, thetadot)
        
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot

    def get_fixed_pts_org(self):
        return [(0.,0.)]
    
    def get_topology(self):
        thr = 0
        a = self.params[0].numpy()

        if a < thr:
            topos = [topo_attr_spiral]
        elif a > thr: # assert threshold
            topos = [topo_rep_spiral, topo_period_attr]
        else:
            topos = [topo_degenerate]
        return topos
    
    def get_dist_from_bifur(self):
        r = self.params[0].numpy()
        return np.linalg.norm(r)

    @staticmethod
    def get_bifurcation_curve(n_points=100):
        a = np.full(n_points, 0)
        omega_min = SimpleOscillator.recommended_param_ranges[1][0]; omega_max = SimpleOscillator.recommended_param_ranges[1][1]
        omega = np.linspace(omega_min, omega_max, n_points)
        return [pd.DataFrame({r'$a$': a, r'$\omega$': omega})]


class InfinitePeriod(FlowSystemODE):
    """
    Simple oscillator:

        rdot = r * (a - r^2)
        thetadot = mu - sin(theta)

    Where:
        - a - radius param (setting non-negative)
        - r - the radius of (x,y)
        - theta - the angle of (x,y)
    
    Strogatz p.262
    """

    min_dims = [-1.,-1.]
    max_dims = [1.,1.]
    
    recommended_param_ranges=[[0.0,0.5], [-2, 2]]
    recommended_param_groups=[recommended_param_ranges]
                    
    eq_string = r'$\dot{x} = r(%.02f - r^2); \dot{\theta} = %.02f - sin(\theta)$'
    short_name='ip'
    
    bifurp_desc = r'$\mu$'
    param_descs = [r'$a$', r'$\mu$']
    

    def forward(self, t, z, **kwargs):
        x = z[..., 0]
        y = z[..., 1]

        a = self.params[0]
        mu = self.params[1]

        r,theta = cartesian_to_polar(x, y)
        
        rdot = r * (a - r ** 2)
        thetadot = mu - torch.sin(theta)

        xdot, ydot = polar_derivative_to_cartesian_derivative(r, theta, rdot, thetadot)
        
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot

    def get_fixed_pts_org(self):
        return [(0.,0.)]
    
    def get_topology(self):
        thr = 1
        a,mu = self.params.numpy()

        if np.abs(mu) < thr:
            topos = [topo_attr_spiral, topo_rep_spiral, topo_saddle]
        else: # assert threshold
            topos = [topo_rep_spiral, topo_period_attr]
        # else:
        #     topos = [topo_degenerate]
        return topos
    
    def get_dist_from_bifur(self):
        r = np.abs(np.abs(self.params[1].numpy()) - 1)
        return np.linalg.norm(r)

    @staticmethod
    def get_bifurcation_curve(n_points=100):
        mu = np.full(n_points, 1)
        a_min = InfinitePeriod.recommended_param_ranges[0][0]; 
        a_max = InfinitePeriod.recommended_param_ranges[0][1]
        a = np.linspace(a_min, a_max, n_points)
        return [pd.DataFrame({r'$a$': a, r'$\mu$': mu})]



class SupercriticalHopf(FlowSystemODE):
    """
    Supercritical Hopf bifurcation:
        rdot = mu * r - r^3
        thetadot = omega + b*r^2
    where:
    - mu controls stability of fixed point at the origin
    - omega controls frequency of oscillations
    - b controls dependence of frequency on amplitude

    Strogatz, p.250
    """

    min_dims = [-1,-1]
    max_dims = [1,1]
    recommended_param_ranges=[[-1.,1.],[-1.,1.],[-1.,1.]]
    recommended_param_groups=[
                                [[-1.,0.],[-1.,1.],[-1.,1.]],[[0.,1.],[-1.,1.],[-1.,1.]]
                             ]
    
    short_name='suphopf'
    bifurp_desc = r'$\mu$'
    # eq_string = r'$\dot{x}_0 = x_1; \dot{x}_1=%.02f x_1 + x_0 - x_0^2 + x_0x_1$'
    param_descs = [r'$\mu$', r'$\omega$', r'$b$']

    def forward(self, t, z, **kwargs):
        x = z[..., 0]
        y = z[..., 1]
        mu = self.params[0]
        omega = self.params[1]
        b = self.params[2]
        r,theta = cartesian_to_polar(x, y)
        
        rdot = mu * r - r**3
        thetadot = omega + b*r**2
        
        xdot, ydot = polar_derivative_to_cartesian_derivative(r, theta, rdot, thetadot)
        
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot

    def get_fixed_pts_org(self):
        return [(0.,0.)]
    
    def get_topology(self):
        mu = self.params[0].numpy()
        if mu < 0:
            topos = [topo_attr_spiral]
        elif mu > 0:
            topos = [topo_rep_spiral, topo_period_attr]
        else:
            topos = [topo_degenerate]

        return topos
    
    def get_dist_from_bifur(self):
        mu = self.params[0].numpy()
        return np.linalg.norm(mu)

    @staticmethod
    def get_bifurcation_curve(n_points=100):
        mu = np.full(n_points, 0)
        omega_min = SupercriticalHopf.recommended_param_ranges[1][0]; omega_max = SupercriticalHopf.recommended_param_ranges[1][1]
        omega = np.linspace(omega_min, omega_max, n_points)
        b_min = SupercriticalHopf.recommended_param_ranges[2][0]; b_max = SupercriticalHopf.recommended_param_ranges[2][1]
        b = np.linspace(b_min, b_max, n_points)
        return [pd.DataFrame({r'$\mu$': mu, 
                             r'$\omega$': omega, 
                             r'$b$': b})]

########################################################## more complex systems ######################################################################


class PreyPredator(FlowSystemODE):
    """
    Rosenzweig-MacArthur model of predator-prey dynamics:
        Fdot = r F (1 - F/K) - a F C / (1 + a h F)
        Cdot = e a F C / (1 + a h F) - mu C
    where:
    - F is prey
    - C is predator
    - r is prey growth rate
    - K is prey carrying capacity
    - a is attack rate
    - h is handling time
    - e is predator growth rate proportional to prey consumption
    - mu is predator death rate

    https://staff.fnwi.uva.nl/a.m.deroos/projects/QuantitativeBiology/43-HopfPoint-Rosenzweig.html
    """

    min_dims = [0,0] 
    max_dims = [0.3,0.3 ]

    # recommended_param_ranges=[[0.1,0.6], [0.2, 0.3], [5.0,5.0], [3.,3.], [0.5, 0.5], [0.1,0.1]]
    recommended_param_ranges=[[0.1,0.6], [0.2, 0.3]]
    recommended_param_groups=[recommended_param_ranges]
    
    short_name='pp'
    n_params = 2#6
    bifurp_desc = ''
    # param_descs = [r'$r$', r'$K$', r'$a$', r'$h$', r'$e$', r'$\mu$']
    param_descs = [r'$r$', r'$K$']
    labels = ['F', 'C']
    plot_param_idx = [0,1]
    a = 5.0 
    h = 3. 
    e = 0.5
    mu = 0.1

    def forward(self, t, z, **kwargs):
        F = z[..., 0] 
        C = z[..., 1] 

        r, K = self.params.numpy()
        a = PreyPredator.a
        h = PreyPredator.h
        e = PreyPredator.e
        mu = PreyPredator.mu

        Fdot = r * F * (1 - F/K) - a * F * C / (1 + a * h * F)
        Cdot = e * a * F * C / (1 + a * h * F) - mu * C
        
        zdot = torch.cat([Fdot.unsqueeze(-1), Cdot.unsqueeze(-1)], dim=-1)
        zdot = torch.cat([Fdot.unsqueeze(-1), Cdot.unsqueeze(-1)], dim=-1)
        
        return zdot

    def get_fixed_pts_org(self):
        # r, K, a, h, e, mu = self.params.numpy()
        r, K = self.params.numpy()
        a = PreyPredator.a
        h = PreyPredator.h
        e = PreyPredator.e
        mu = PreyPredator.mu
        
        # handling only fixed pt where neither F nor C is zero
        F_st = 1 / (e * a / mu - a*h)
        C_st = (r/a * (1 - F_st/K)) * (1+ a * h * F_st) 
        
        return [(F_st, C_st)]
    
    @staticmethod
    def J_helper(F, C, r, K):
        
        a = PreyPredator.a
        h = PreyPredator.h
        e = PreyPredator.e
        mu = PreyPredator.mu
        
        return np.array([[r * (1 - 2*F/K) - a * C / (1 + a * h * F) + a**2 * h * F * C / (1 + a * h * F)**2, 
                         -a * F / (1 + a * h * F)],
                            [e * a * C / (1 + a * h * F) - e * a**2 * h * C * F / (1 + a * h * F)**2 , 
                            e * a * F / (1 + a * h * F) - mu]])
    

    def J(self, F, C):
        r,K = self.params.numpy()
        return PreyPredator.J_helper(F, C, r,K)
    
    def get_topology(self):
        return self.get_topology_supercriticalhopf()
    
    def get_dist_from_bifur(self):
        pass

    @staticmethod
    def get_bifurcation_curve(n_points=100):
        r_min = PreyPredator.recommended_param_ranges[0][0]
        r_max = PreyPredator.recommended_param_ranges[0][1]
        r = np.linspace(r_min, r_max ,n_points)

        a = PreyPredator.a
        h = PreyPredator.h
        e = PreyPredator.e
        mu = PreyPredator.mu
        
        def equation_trace(K,r):
            F_st = 1 / (e * a / mu - a*h)
            C_st = (r/a * (1 - F_st/K)) * (1+ a * h * F_st) 
            # tr = r * (1 - 2*F_st/K) - a * C_st / (1 + a * h * F_st) + a**2 * h * F_st * C_st / (1 + a * h * F_st)**2 + e * a * F_st / (1 + a * h * F_st) - mu
            tr = np.trace(PreyPredator.J_helper(F_st, C_st, r, K))
            return tr

        K0 = np.full(n_points, .26)

        # Solve the equation using fsolve
        K = fsolve(equation_trace, K0, args=(r))[0]

        df = pd.DataFrame({r'$r$': r, r'$K$': K})
        # filter saddle nodes by checking determinant > 0

        def equation_det(K,r):
            F_st = 1 / (e * a / mu - a*h)
            C_st = (r/a * (1 - F_st/K)) * (1+ a * h * F_st) 
            
            J_st = PreyPredator.J_helper(F_st, C_st, r, K)
            det = J_st[0,0] * J_st[1,1] - J_st[0,1] * J_st[1,0]
            return det
        
        df = df[equation_det(K,r) > 0]

        K_min = PreyPredator.recommended_param_ranges[1][0]
        K_max = PreyPredator.recommended_param_ranges[1][1]
        df = df[(df[r'$K$'] >= K_min) & (df[r'$K$'] <= K_max)]

        return [df]
        




class BZreaction(FlowSystemODE):
    """
    BZ reaction (undergoing Hopf bifurcation):
        xdot = a - x - 4*x*y / (1 + x^2)
        ydot = b * x * (1 - y / (1 + x^2))
    where:
        - a, b depend on empirical rate constants and on concentrations of slow reactants 

    Strogatz, p.256
    """
    
    min_dims = [0,0] 
    max_dims = [10,20] # bzreaction2
    # max_dims = [5,17] #bzreaction_new
    
    recommended_param_ranges=[[3,19], [2, 6]]
    recommended_param_groups=[recommended_param_ranges]
    
    short_name='bz'
    bifurp_desc = r'$(a, \frac{3a}{5} - \frac{25}{a})$'
    param_descs = [r'$a$', r'$b$']

    # foldx = 5
    # foldy = 10
    # eq_string = r'$\dot{x}_0 = x_1; \dot{x}_1=%.02f x_1 + x_0 - x_0^2 + x_0x_1$'

    # def __init__(self, params,  min_dims=None, max_dims=None, foldx=5, foldy=10, **kwargs):
    #     min_dims = self.min_dims if min_dims is None else min_dims
    #     max_dims = self.max_dims if max_dims is None else max_dims
    #     super().__init__(params=params, min_dims=min_dims, max_dims=max_dims, **kwargs)
    #     self.foldx = foldx
    #     self.foldy = foldy
    #     self.shiftx = min(0, self.min_dims[0])
    #     self.shifty = min(0, self.min_dims[1])
    

    def forward(self, t, z, **kwargs):
        x = z[..., 0] 
        y = z[..., 1] 

        # x = self.foldx * (x - self.shiftx)
        # y = self.foldy * (y - self.shifty)

        a,b = self.params.numpy()
        
        xdot = a - x - 4*x*y / (1 + x**2)
        ydot = b * x * (1 - y / (1 + x**2))
                
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        
        return zdot

    def get_fixed_pts_org(self):
        a,b = self.params.numpy()
        
        x_st = a / 5
        y_st = 1 + (a/5)**2
        
        return [(x_st, y_st)]
    
    def J(self, x, y):
        a, b = self.params.numpy()
        # xdot = a - x - 4*x*y / (1 + x**2)
        # ydot = b * x * (1 - y / (1 + x**2))
        
        return np.array([[-1 + 4*y*(x**2-1) / (1 + x**2)**2 # - 8*x**2*y / (1 + x**2)**2
                         , -4*x / (1 + x**2)], 
                         [b*y*(x**2 - 1) / (1 + x**2)**2 + b, #b * (1 - y / (1 + x**2)) + b * x * (y * 2 * x / (1 + x**2)**2), 
                         -b * x / (1 + x**2)]])
        # return 1 / (1 + x**2) * np.array([[3*x**2  - 5, -4*x],[2*b*x**2, -b*x]])
    
    def get_topology(self):
        return self.get_topology_supercriticalhopf()
    
    def get_dist_from_bifur(self):
        f = lambda a: 3*a/5 - 25/a
        a_limits = (eps, 1e4)
        a = self.params[0].numpy()
        b = self.params[1].numpy()
        return get_dist_from_bifur_curve(a, b, f=f, a_limits=a_limits)

    @staticmethod
    def get_bifurcation_curve(n_points=100):
        f = lambda a: 3*a/5 - 25/a
        a_min = BZreaction.recommended_param_ranges[0][0]
        a_max = BZreaction.recommended_param_ranges[0][1]
        a = np.linspace(a_min, a_max ,n_points)
        df = pd.DataFrame({r'$a$': a, r'$b$': f(a)})
        b_min = BZreaction.recommended_param_ranges[1][0]
        b_max = BZreaction.recommended_param_ranges[1][1]
        df = df[(df[r'$b$'] >= b_min) & (df[r'$b$'] <= b_max)]
        return [df]


class Selkov(FlowSystemODE):
    """
    Selkov oscillator:
        xdot = -x + ay + x^2y
        ydot = b - ay - x^2y

    Strogatz, p. 209
    """
    min_dims = [0,0]
    max_dims = [3,3]
    # max_dims = [3.5,4]
    # # recommended_param_ranges=[[.01,.11],[.02,1.2]] 
    recommended_param_ranges=[[.01,.11],[.02,1.2]] # cutting off param regime..

    # selkov2
    # min_dims = [0.01, 0.0]
    # max_dims = [2, 3.6]
    # recommended_param_ranges=[[.02,.11],[.02,1.2]] 

    # min_dims = [0.01, 0.0]
    # max_dims = [2, 2.9]
    # recommended_param_ranges=[[.03,.11],[.02,1.2]] 

    # TODO: changed!
    # "bottom"
    # min_dims = [0,0]
    # max_dims = [3.5,4]
    # # recommended_param_ranges=[[0.01,1.],[0.01,2.5]] # trying 
    # recommended_param_ranges=[[.01,.11],[.02,0.6]] # cutting off param regime..

    # total range
    # min_dims = [0., 0. ]
    # max_dims = [1.25, 5.1]
    # recommended_param_ranges=[[.01,.11],[.02,1.2]]

    # # zoom param
    # "top"
    # min_dims = [0.6, 0.7 ] # top_new
    # max_dims = [1.25, 1.65]
    # min_dims = [0.3, 0.3 ] #top2
    # max_dims = [1.7, 2]
    # recommended_param_ranges=[[.01,.11],[0.6,1.2]]

    recommended_param_groups = [recommended_param_ranges]
    # eq_string=r'$\dot{x}_0 = -x_0 + {0:%.02f} x_1 + x_0^2 x_1; \dot{x}_1= {1:%.02f} - {0:%.02f} x_1 - x_0x_1$'
    short_name='sl'
    bifurp_desc = r'$(a,\sqrt{\frac{1}{2} (1-2a \pm \sqrt{1-8a})})$'
    param_descs = [r'$a$', r'$b$']

    
    # def __init__(self, params,  min_dims=[-1.,-1.], max_dims=[1.,1.], foldx=1.5, foldy=1.5, shiftx=-1, shifty=-1, **kwargs): # TODO: changed! foldx=1.5, foldy=1.5
    #     super().__init__(params=params, min_dims=min_dims, max_dims=max_dims, foldx=foldx, foldy=foldy, shiftx=shiftx, shifty=shifty, **kwargs)
        
    def forward(self, t, z, **kwargs):
        x = z[...,0]
        y = z[...,1]
        # x = self.foldx * (x - self.shiftx)
        # y = self.foldy * (y - self.shifty)

        a, b = self.params
        # x = (x + 1) * 1.5
        # y = (y + 1) * 1.5
        xdot = -x + a*y + x**2*y
        ydot = b - a*y - x**2*y
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot

    def get_polynomial_representation(self):
        dx = pd.DataFrame(0.0, index=['selkov'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['selkov'], columns=self.polynomial_terms)
        
        a, b = self.params.numpy()
        
        dx.loc['selkov']['$x_0$'] = -1
        dx.loc['selkov']['$x_1$'] = a
        dx.loc['selkov']['$x_0^2x_1$'] = 1

        dy.loc['selkov']['1'] = b
        dy.loc['selkov']['$x_1$'] = -a
        dy.loc['selkov']['$x_0^2x_1$'] = -1

        return dx, dy

    def get_fixed_pts_org(self):
        a,b = self.params.numpy()
        
        x_st = b
        y_st = b / (a + b**2)
        return [(x_st, y_st)]
    
    def J(self, x, y):
        a = self.params[0].numpy()
        return np.array([[-1 + 2*x*y, a + x**2], [-2*x*y, -a - x**2]])

    def get_topology(self):
        return self.get_topology_supercriticalhopf()
    

    def get_dist_from_bifur(self):
        f_plus = lambda a: np.sqrt(1/2 * (1-2*a + np.sqrt(1-8*a)))
        f_minus = lambda a: np.sqrt(1/2 * (1-2*a - np.sqrt(1-8*a)))
        a_limits = (eps, 1/8 - eps)
        a = self.params[0].numpy()
        b = self.params[1].numpy()
        bifurp_plus = get_dist_from_bifur_curve(a, b, f=f_plus, a_limits=a_limits)
        bifurp_minus = get_dist_from_bifur_curve(a, b, f=f_minus, a_limits=a_limits)
        min_dist = np.minimum(np.abs(bifurp_plus), np.abs(bifurp_minus))
        return min_dist
    
    @staticmethod
    def get_bifurcation_curve(n_points=100):
        a_min = Selkov.recommended_param_ranges[0][0]; a_max = Selkov.recommended_param_ranges[0][1]; 
        a = np.linspace(a_min, a_max, n_points); 
        f_plus = lambda a: np.sqrt(1/2 * (1-2*a + np.sqrt(1-8*a)))
        f_minus = lambda a: np.sqrt(1/2 * (1-2*a - np.sqrt(1-8*a)))
        dfs = [pd.DataFrame({r'$a$':a, r'$b$':f_plus(a)}), pd.DataFrame({r'$a$':a, r'$b$':f_minus(a)})]
        b_min = Selkov.recommended_param_ranges[1][0]; b_max = Selkov.recommended_param_ranges[1][1]; 
        for df in dfs:
            df = df[(df[r'$b$'] >= b_min) & (df[r'$b$'] <= b_max)]  
        return dfs




class VanDerPol(FlowSystemODE):
    """
    Van der pol oscillator:
        xdot = y
        ydot = mu * (1-x^2) * y - x

    Strogatz, p. 198


    Rescaled version
        xdot = y
        ydot = mu*y -x^2*y - x
    """
    min_dims = [-3., -3.]
    max_dims = [3., 3.]
    # recommended_param_ranges=[[-.5,.5]]
    recommended_param_ranges = [[-1,1]]
    # recommended_param_ranges=[[-.05,.05]] #[[.005,.05]] # TODO:  Watch out with range (changed from 0.05 max)
    recommended_param_groups=[recommended_param_ranges]

    # recommended_param_ranges=[[.005,.05]]
    # recommended_param_groups=[[[.005,.05]]]

    eq_string=r'$\dot{x}_0=x_1; \dot{x}_1 = %.02f (1-x_0^2)x_1 - x_0$'
    short_name='vp'

    bifurp_desc = r'$\mu$'
    param_descs = [r'$\mu$']
    plot_param_idx = [0,0]

    
    def forward(self, t, z, **kwargs):

        x = z[...,0]
        y = z[...,1]

        mu = self.params[0]
        
        xdot = y
        ydot = mu * y - x - x**2*y
        
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot

    def get_polynomial_representation(self):
        dx = pd.DataFrame(0.0, index=['vanderpol'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['vanderpol'], columns=self.polynomial_terms)
        mu = self.params.numpy()[0]
        dx.loc['vanderpol']['$x_1$'] = 1
        dy.loc['vanderpol']['$x_1$'] = mu
        dy.loc['vanderpol']['$x_0^2x_1$'] = -1
        dy.loc['vanderpol']['$x_0$'] = -1 
        return dx, dy
        
    def get_fixed_pts_org(self):
        return [(0.,0.)]
    
    def J(self, x, y):
        mu = self.params[0].numpy()
        return np.array([[0, 1], [-1 - 2*x*y, mu - x**2]])
    
    def get_topology(self):
        return self.get_topology_supercriticalhopf()

    def get_dist_from_bifur(self):
        mu = self.params[0].numpy()
        return np.linalg.norm(mu)

    @staticmethod
    def get_bifurcation_curve(n_points=100):
        mu = np.full(n_points, 0)
        return [pd.DataFrame({r'$\mu$': mu})]



class BiasedVanDerPol(FlowSystemODE):
    """
    Biased Van der pol oscillator:
    
        xdot = y
        ydot = mu * (1-x^2) * y - x + a

    Strogatz Ex 8.2.1 or https://demonstrations.wolfram.com/HopfBifurcationInABiasedVanDerPolOscillator/

    """
    min_dims = [-5., -5.]
    max_dims = [5., 5.]
    recommended_param_ranges = [[0,1],[-2,2]]
    recommended_param_groups=[recommended_param_ranges]

    eq_string=r'$\dot{x}_0=x_1; \dot{x}_1 = %.02f (1-x_0^2)x_1 - x_0 + %.02f $'
    short_name='bvp'

    bifurp_desc = r'$a$'
    param_descs = [r'$\mu$', r'$a$']
    plot_param_idx = [1,0]

    def forward(self, t, z, **kwargs):

        x = z[...,0]
        y = z[...,1]

        mu,a = self.params.numpy()
        
        xdot = y
        ydot = mu * y - x - x**2*y + a
        
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot

    def get_polynomial_representation(self):
        dx = pd.DataFrame(0.0, index=['biased_vanderpol'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['biased_vanderpol'], columns=self.polynomial_terms)
        mu,a = self.params.numpy()
        dx.loc['biased_vanderpol']['$x_1$'] = 1
        dy.loc['biased_vanderpol']['$x_1$'] = mu
        dy.loc['biased_vanderpol']['$x_0^2x_1$'] = -mu
        dy.loc['biased_vanderpol']['$x_0$'] = -1
        dy.loc['biased_vanderpol']['1'] = a
        return dx, dy
        
    def get_fixed_pts_org(self):
        a = self.params[1].numpy()
        return [(a,0.)]
    
    def J(self, x, y):
        mu = self.params[0].numpy()
        return np.array([[0, 1], [-1 - 2*x*y, mu - x**2]])
    
    def get_topology(self):
        mu,a = self.params.numpy()
        if mu < 0:
            raise ValueError('mu must be positive, otherwise subcritical Hopf bifurcation')
        return self.get_topology_supercriticalhopf()

    def get_dist_from_bifur(self):
        a = self.params[1].numpy()
        thr = 1
        return np.linalg.norm(np.abs(a) - thr)

    @staticmethod
    def get_bifurcation_curve(n_points=100):
        mu_min = BiasedVanDerPol.recommended_param_ranges[0][0]
        mu_max = BiasedVanDerPol.recommended_param_ranges[0][1]
        mu = np.linspace(mu_min, mu_max ,n_points)
        a_plus = np.full(n_points, 1)
        a_minus = np.full(n_points, -1)
        dfs = [pd.DataFrame({r'$\mu$':mu, r'$a$':a_plus}), 
               pd.DataFrame({r'$\mu$':mu, r'$a$':a_minus})]
        return dfs



class FitzhughNagumo(FlowSystemODE):
    """
    Fitzhugh Nagumo model (a prototype of an excitable system, e.g. neuron)

        udot = u - u^3 / 3. - w + I # TODO: fix
        wdot = 1/tau * (u + a - b * w)

    Where:
        - I - external stimulus
        - tau - time constant
        - a - 
        - b - 

    https://demonstrations.wolfram.com/PhasePlaneDynamicsOfFitzHughNagumoModelOfNeuronalExcitation/
    """

    labels = ['u', 'w'] # TODO: can the voltage be negative???
    n_params = 4
    params = torch.zeros(n_params)
    # min_dims = [0.,0.]
    # max_dims = [1.,1.]
    min_dims = [-2, -2]
    max_dims = [2.0, 2.0]

    # recommended_param_ranges = [[0,.7],[12.5,12.6],[.7,.8],[.8,.9]] # TODO: check no other bifurcation in this regime
    recommended_param_ranges = [[0,.5],[12.5,12.5],[.7,.7],[.5,.9]]
    recommended_param_groups=[recommended_param_ranges]
    # recommended_param_groups = [[[0,.35],[12.5,12.6],[.7,.8],[.8,.9]] ,[[.35,.7],[12.5,12.6],[.7,.8],[.8,.9]] 
    #                            ]

    eq_string=r'$\dot{x}_0 = %.02f + x_0  - x_1 - x_0^3/3; \dot{x}_1 = \frac{1}{%.02f}(x_0 + %.02f - %.02f x_1)$'
    short_name='fn'
    
    param_descs = [r'$I$', r'$\tau$', r'$a$', r'$b$']
    bifurp_desc = ''
    plot_param_idx = [0,3]

    # def __init__(self, params=params, labels=labels, min_dims=min_dims, max_dims=max_dims, **kwargs):
    #     super().__init__(params, labels, min_dims=min_dims, max_dims=max_dims, **kwargs)

    def forward(self, t, z, **kwargs):
        u = z[..., 0]
        w = z[..., 1]

        I, tau, a, b = self.params.numpy()

        # u = self.foldx * u
        # w = self.foldy * w
        
        udot = u - u**3 / 3. - w + I
        wdot = (1. / tau) * (u + a - b*w)
        zdot = torch.cat([udot.unsqueeze(-1), wdot.unsqueeze(-1)], dim=-1)

        return zdot
    
    @staticmethod
    def get_fixed_pts_org_helper(I, a, b, verbose=False):
        
        # def equation(x, I, a, b):
        #     return x - x**3 / 3. + I - (x + a)/b
        
        # # Initial guess for x
        # u0 = 1.0

        # # Solve the equation using fsolve
        # u = fsolve(equation, u0, args=(I, a, b))[0]

        # if verbose:
        #     print("Solution for u:", u)

        # w = u - u**3 / 3. + I

        # if verbose:
        #     print("Solution for w:", w)

        # from wolfram alpha, b\neq0
        v1 = np.power(-81*a*b**2 + np.sqrt((81*b**3*I - 81*a*b**2)**2 - 2916*(b-1)**3*b**3) + 81*b**3*I, 1/3)
        v2 = 3*np.power(2, 1/3)
        u = v2 * (b-1) / v1 + v1 / (v2 * b)

        w = u - u**3 / 3. + I

        return [(u,w)] # TODO: what if at regime with 3 interactions?

    def get_fixed_pts_org(self, verbose=False):
        I, tau, a, b = self.params.numpy()
        return FitzhughNagumo.get_fixed_pts_org_helper(I, a, b)
    
    @staticmethod
    def J_helper(u, w, I, tau, a, b):
        return np.array([[1 - u**2, -1], [1. / tau, -b/tau]])
    
    def J(self, u, w):
        I, tau, a, b = self.params.numpy()
        return FitzhughNagumo.J_helper(u, w, I, tau, a, b)
    
    def get_topology(self):
        return self.get_topology_supercriticalhopf()
    
    def get_polynomial_representation(self):
        """
        Considering v as x and w as y, return the polynomial representation of the system.
        """
        dx = pd.DataFrame(0.0, index=['fitzhughnagumo'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['fitzhughnagumo'], columns=self.polynomial_terms)
        
        I, tau, a, b = self.params.numpy()
        dx.loc['fitzhughnagumo']['1'] = I
        dx.loc['fitzhughnagumo']['$x_0$'] = 1
        dx.loc['fitzhughnagumo']['$x_1$'] = -1
        dx.loc['fitzhughnagumo']['$x_0^3$'] = -1/3
        dy.loc['fitzhughnagumo']['$x_0$'] = (1/tau)
        dy.loc['fitzhughnagumo']['1'] = (1 / tau) * a
        dy.loc['fitzhughnagumo']['$x_1$'] = (1/tau) * (-b)
        
        return dx, dy
    
    def get_dist_from_bifur(self):
        I, tau, a, b = self.params.numpy()
        u_st, w_st = self.get_fixed_pts_org(verbose=False)[0]
        return np.linalg.norm(1 - u_st**2 - b/tau)
    
    @staticmethod
    def get_bifurcation_curve(n_points=100):
        if FitzhughNagumo.plot_param_idx != [0,3]:
            raise ValueError('Only implemented for plot_param_idx = [0,3]')
        b_min = FitzhughNagumo.recommended_param_ranges[3][0]
        b_max = FitzhughNagumo.recommended_param_ranges[3][1]
        b = np.linspace(b_min, b_max ,n_points)

        tau = FitzhughNagumo.recommended_param_ranges[1][0] # const
        a = FitzhughNagumo.recommended_param_ranges[2][0] # const

        def equation_trace(I, a, b, tau):
            u_st, w_st = FitzhughNagumo.get_fixed_pts_org_helper(I, a, b)[0]
            tr = np.trace(FitzhughNagumo.J_helper(u_st, w_st, I, tau, a, b))
            return tr
        
        I0 = np.full(n_points, 0.5)
        I = fsolve(equation_trace, I0, args=(a, b, tau))[0]

        df = pd.DataFrame({r'$I$':I, r'$\tau$':tau, r'$a$':a, r'$b$':b})
        
        # filter determinant >0 (stable)
        def equation_det(b, I, a, tau):
            u_st, w_st = FitzhughNagumo.get_fixed_pts_org_helper(I, a, b)[0]
            J_st = FitzhughNagumo.J_helper(u_st, w_st, I, tau, a, b)
            det = J_st[0,0] * J_st[1,1] - J_st[0,1] * J_st[1,0]
            return det
        
        df = df[equation_det(b, I, a, tau) > 0]
        
        I_min = FitzhughNagumo.recommended_param_ranges[0][0]
        I_max = FitzhughNagumo.recommended_param_ranges[0][1]
        df = df[(df[r'$I$'] >= I_min) & (df[r'$I$'] <= I_max)]
        return [df]
    
    # @staticmethod
    # def get_bifurcation_curve(n_points=100):
        
    #     # I_min = FitzhughNagumo.recommended_param_ranges[0][0]
    #     # I_max = FitzhughNagumo.recommended_param_ranges[0][1]
    #     # I = np.linspace(I_min, I_max ,n_points)
    #     tau = FitzhughNagumo.recommended_param_ranges[1][0] # const
    #     a = FitzhughNagumo.recommended_param_ranges[2][0] # const
    #     # a_min = FitzhughNagumo.recommended_param_ranges[2][0]
    #     # a_max = FitzhughNagumo.recommended_param_ranges[2][1]
    #     # a = np.linspace(a_min, a_max ,n_points)

    #     b_min = FitzhughNagumo.recommended_param_ranges[3][0]
    #     b_max = FitzhughNagumo.recommended_param_ranges[3][1]
    #     b = np.linspace(b_min, b_max ,n_points)

    #     def equation(x, a, b, tau): # x corresponds to I
    #         fx = (-81*a*b**2 + np.sqrt((81*b**3*x - 81*a*b**2)**2 - 2916*(b-1)**3*b**3) + 81*b**3*x)**(1/3)
    #         return (3*2**(1/3) * (b-1) / fx + fx / (3*2**(1/3) * b))**2 - (1 - b/tau)

    #     I0 = np.full(n_points, 0.5)

    #     # Solve the equation using fsolve
    #     I = fsolve(equation, I0, args=(a, b, tau))[0]

    #     df = pd.DataFrame({r'$I$':I, r'$\tau$':tau, r'$a$':a, r'$b$':b})
    #     I_min = FitzhughNagumo.recommended_param_ranges[0][0]
    #     I_max = FitzhughNagumo.recommended_param_ranges[0][1]
    #     df = df[(df[r'$I$'] >= I_min) & (df[r'$I$'] <= I_max)]
    #     return [df]
    
########################################################## lienard ############################################################
        

class Lienard(FlowSystemODE):
    """
    A general class of oscillators where:
        xdot = y
        ydot = -g(x) -f(x)*y

    And:
    - $f,g$ of polynomial basis (automatically then continuous and differentiable for all x)
    - $g$ is an odd function ($g(-x)=-g(x)$)
    - $g(x) > 0$ for $x>0$
    - cummulative function of f, $F(x)=\int_0^xf(u)du$, and is negative for $0<x<a, F(x)=0$, $x>a$ is positive and nondecreasing

    """
    def __init__(self, params, flip=False, **kwargs):
        super().__init__(params=params, **kwargs)
        
        self.flip = flip

    def forward(self, t, z):
        # x = self.foldx * z[...,0]
        # y = self.foldy * z[...,1]
        
        x = z[...,0]
        y = z[...,1]
        
        xdot = y
        ydot = -self.g(x) -self.f(x)*y
        if self.flip:
            xdot = -self.g(y) -self.f(y)*x
            ydot = x
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)

        return zdot
    
    def get_info(self):
        return super().get_info(exclude=super().exclude + ['f', 'g'])


class LienardPoly(Lienard):
    """
    Lienard oscillator with polynomial $f,g$ up to degree 3:
        f(x) = c + d x^2
        g(x) = a x + b x^3
    where c < 0 and a,b,d > 0 there is a limit cycle according to Lienard equations. 
    Here we let c be positive to allow for a fixed point. 
    """

    recommended_param_ranges=[[0.0,1.0], [0.5,0.5], [-1.0,1.0], [0.5,0.5]]
    recommended_param_ranges=[[0.0,1.0], [1.0,1.0], [-1.0,1.0], [1.0,1.0]]
    # recommended_param_ranges=[[0.0,1.0], [0.0,1.0], [-1.0,1.0], [0.0,1.0]] # TODO: freezed b,d values
    # I want to test both c changing from negative to positive and a changing from positive to negative
    recommended_param_groups = [recommended_param_ranges]
    param_descs = [r'$a$', r'$b$', r'$c$', r'$d$']
    bifurp_desc = r'$c'
    plot_param_idx = [2,0]
    min_dims = [-4.2, -4.2]
    max_dims = [4.2, 4.2]

    def __init__(self, params, **kwargs):
        super().__init__(params=params, **kwargs)
        a,b,c,d = self.params
        # assert(self.params[[0,1]].sum() > 0)
        self.g = lambda x: a * x + b * x**3

        # assert(c < 0 and d > 0) # allowing negative c
        self.f = lambda x: c + d * x**2
        # self.foldx = 4.2 ###8 # 
        # self.foldy = 4.2 ###8 # 
    
    def get_fixed_pts_org(self):
        return [(0.,0.)]

    def J(self, x, y):
        a,b,c,d = self.params.numpy()
        f_der = lambda x: 2 * d * x
        g_der = lambda x: a + 3 * b * x**2

        return np.array([[0, 1], [-g_der(x) -f_der(x)*y, -self.f(x)]])
    
    def get_topology(self):
        return self.get_topology_supercriticalhopf()
    
    def get_dist_from_bifur(self):
        c = self.params[2].numpy()
        return np.linalg.norm(c)
    
    @staticmethod
    def get_bifurcation_curve(n_points=100):
        a_min = LienardPoly.recommended_param_ranges[0][0]; a_max = LienardPoly.recommended_param_ranges[0][1]
        a = np.linspace(a_min, a_max, n_points)
        
        b_min = LienardPoly.recommended_param_ranges[1][0]; b_max = LienardPoly.recommended_param_ranges[1][1]
        b = np.linspace(b_min, b_max, n_points)
        
        c = np.full(n_points, 0)

        d_min = LienardPoly.recommended_param_ranges[2][0]; d_max = LienardPoly.recommended_param_ranges[2][1]
        d = np.linspace(d_min, d_max, n_points)
        return [pd.DataFrame({r'$a$': a, 
                             r'$b$': b, 
                             r'$c$': c, 
                             r'$d$': d})]

    def get_polynomial_representation(self):
        """
        Considering v as x and w as y, return the polynomial representation of the system.
        """
        dx = pd.DataFrame(0.0, index=['lienard_poly'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['lienard_poly'], columns=self.polynomial_terms)
        
        a,b,c,d = self.params.numpy()
        
        dx.loc['lienard_poly']['$x_1$'] = 1
        
        dy.loc['lienard_poly']['$x_0$'] = -1
        dy.loc['lienard_poly']['$x_0^3$'] = -b
        dy.loc['lienard_poly']['$x_1$'] = -c
        dy.loc['lienard_poly']['$x_0^2x_1$'] = -d
        
        return dx, dy

class LienardSigmoid(Lienard):
    """
    Lienard oscillator with polynomial $f$ and sigmoid $g$:
        f(x) = b + c x^2
        g(x) = 1 / (1 + e^(-ax)) - 0.5
    where b < 0 and a,c > 0 there is a limit cycle according to Lienard equations. 
    Here we let b be positive to allow for a fixed point. 
    """
    recommended_param_ranges=[[1.0,2.0], [-1.0,1.0], [1.0,1.0]] # a,b,c
    # recommended_param_ranges=[[0.0,1.0], [-1.0,1.0], [1.0,1.0]] # a,b,c
    # recommended_param_ranges=[[0.0,1.0], [-1.0,1.0], [0.0,1.0]] # TODO: changed! c to consta,b,c
    recommended_param_groups = [recommended_param_ranges]

    param_descs = [r'$a$', r'$b$', r'$c$'] 
    bifurp_desc = r'$b$'
    plot_param_idx = [1,0]

    # min_dims = [-2.5, -2.5]
    # max_dims = [2.5, 2.5]
    
    min_dims = [-1.5, -1.5]
    max_dims = [1.5, 1.5]
    

    def __init__(self, params, **kwargs):
        super().__init__(params=params, **kwargs)
        assert(self.params[0] > 0)
        self.g = lambda x: 1 / (1 + torch.exp(-self.params[0] * x)) - 0.5
        # self.g = lambda x: self.params[0] * torch.sin(x)

        # assert(self.params[1] < 0 and self.params[2] > 0)
        assert(self.params[2] > 0)
        self.f = lambda x: self.params[1] + self.params[2] * x**2
        # self.foldx = 2.5 ### 4 # 
        # self.foldy = 2.5 ### 4 # 
        
    def get_fixed_pts_org(self):
        return [(0.,0.)]
    
    def J(self, x, y):
        a,b,c = self.params.numpy()
        f_der = lambda x: 2 * c * x
        g_der = lambda x: a * np.exp(-a * x) / (1 + np.exp(-a * x))**2

        return np.array([[0, 1], [-g_der(x) -f_der(x)*y, -self.f(x)]])

    def get_topology(self):
        return self.get_topology_supercriticalhopf()
    
    def get_dist_from_bifur(self):
        b = self.params[1].numpy()
        return np.linalg.norm(b)
    
    @staticmethod
    def get_bifurcation_curve(n_points=100):
        a_min = LienardSigmoid.recommended_param_ranges[0][0]; a_max = LienardSigmoid.recommended_param_ranges[0][1]
        a = np.linspace(a_min, a_max, n_points)
        
        b = np.full(n_points, 0)

        c_min = LienardSigmoid.recommended_param_ranges[2][0]; c_max = LienardSigmoid.recommended_param_ranges[2][1]
        c = np.linspace(c_min, c_max, n_points)
        
        return [pd.DataFrame({r'$a$': a, 
                             r'$b$': b, 
                             r'$c$': c})]






class Polynomial(FlowSystemODE):
    """
    Polynomial system of equations up to poly_order. e.g., for poly_order=2, the system is:
        xdot = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2 
        ydot = d0 + d1*x + d2*y + d3*x^2 + d4*x*y + d5*y^2 
    """
    dim = 2

    min_dims = [-1.,-1.]
    max_dims = [1., 1.]
    
    recommended_param_ranges = 20 * [[-3., 3.]]
    recommended_param_groups = [recommended_param_ranges]
 
    param_descs = [r'$c_0$', r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$', r'$c_5$', r'$d_0$', r'$d_1$', r'$d_2$', r'$d_3$', r'$d_4$', r'$d_5$']
    def __init__(self, params=None, labels=None, min_dims=None, max_dims=None, poly_order=3, include_sine=False, include_exp=False, **kwargs):
        """
        Initialize the polynomial system.
        :param params: correspond to column vector of library terms for dim1, concatenated with column vector library terms for dim2 ()
        :param poly_order: the order of the polynomial system
        """
        # labels = ['x', 'y'] if labels is None else labels
        # min_dims = [-1.0,-1.0] if min_dims is None else min_dims
        # max_dims = [1.0,1.0] if max_dims is None else max_dims
        super().__init__(params=params, labels=labels, min_dims=min_dims, max_dims=max_dims, **kwargs)

        self.poly_order = int(poly_order)
        self.include_sine = include_sine
        self.include_exp = include_exp
        L = self.generate_mesh()
        self.library, self.library_terms = sindy_library(L.reshape(self.num_lattice**self.dim, self.dim), self.poly_order,
                                      include_sine=self.include_sine, include_exp=self.include_exp)
        self.library = self.library.float()
        
    def forward(self, t, z, **kwargs):
        # params = self.params.reshape(-1, self.dim).to(self.device) # Error prone # TODO: changed!!
        params = self.params.reshape(self.dim, -1).T.to(self.device) # Error prone
        z_shape = z.shape
        library = self.get_library(z)
        zdot = torch.einsum('sl,ld->sd', library, params).to(self.device)

        zdot = zdot.reshape(*z_shape)
    
        return zdot

    def params_str(self, s=''):
        """
        Sparse representation of the parameters.
        """
        nterms = len(self.library_terms)
        if (nterms * self.dim)!= len(self.params):
            return s
        eqs = []
        params = self.params.numpy()
        for a in range(self.dim):
            eq = r'$\dot{x_' + f'{a}' + '}' + '= $'
            first_term = True
            for lt, pr in zip(self.library_terms, params[a * nterms:(a + 1) * nterms]):
                if np.abs(pr) > 0:
                    sgn = np.sign(pr)
                    if first_term:
                        conj = '' if sgn > 0 else '-'
                    else:
                        conj = '+' if sgn > 0 else '-'
                    first_term = False
                    eq += f' {conj} {np.abs(pr):.3f}' + lt
            eqs.append(eq)
        
        return s + '\n'.join(eqs)

    def get_polynomial_representation(self):
        nterms = len(self.library_terms)
        dx = pd.DataFrame(0.0, index=['polynomial'], columns=self.library_terms) 
        dy = pd.DataFrame(0.0, index=['polynomial'], columns=self.library_terms)
        params = [p.numpy() for p in self.params]
        for ilb,lb in enumerate(self.library_terms):
            dx[lb] = params[ilb]
            dy[lb] = params[ilb + nterms]
        return dx, dy

########################################################## biological oscillators ############################################################

class Repressilator(FlowSystemODE):
    """
    Elowitz and Leibler repressilator:
        dm_i = -m_i + a/(1 + p_j^n) + a0
        dp_i = -b(p_i - m_i) 
    where:
    - m_i (/p_i) denote mRNA(/protein) concentration of gene i
    - i = lacI, tetR, cI and j = cI, lacI, tetR
    - a0 - leaky mRNA expression
    - a - transcription rate without repression
    - b - ratio of protein and mRNA degradation rates
    - n - Hill coefficient

    see Box 1 in https://www.nature.com/articles/35002125#Sec3

    Params from: https://gist.github.com/AndreyAkinshin/37f3e68a1576f9ea1e5c01f2fd64fe5e

    TODO: go back to https://www.scirp.org/html/2-1100677_85309.htm#f2
    """
    nparams = 2 # going to vary a and b
    dim = 6 # going to look at TetR (GFP repressor) and LacI(TetR repressor) proteins concentrations
    min_dims = dim * [1e-4,]  
    max_dims = dim * [1e3]
    n = 2
    a0 = 0.2 # 0
    # recommended_param_ranges=[[1e-4,1e5], [1e-4, 1e4]] # need to sample uniformly from exponential values
    # p_osc = [2e2,3]
    # p_pt = [1,10]
    recommended_param_ranges=[[1e-4,30], [1e-4, 10]] 
    recommended_param_groups=[recommended_param_ranges]
    
    short_name='rp'
    bifurp_desc = ''
    param_descs = [r'$\alpha$', r'$\beta$']
    labels = ['mLacI','pLacI','mTetR','pTetR','mcI','pcI']
    plot_param_idx = [0,1]

    def __init__(self, params,  min_dims=None, max_dims=None, **kwargs):
        min_dims = self.min_dims if min_dims is None else min_dims
        max_dims = self.max_dims if max_dims is None else max_dims
        super().__init__(params=params, min_dims=min_dims, max_dims=max_dims, labels=self.labels, **kwargs)
    
    def forward(self, t, z, **kwargs):
        mLacI = z[..., 0]
        pLacI = z[..., 1]
        mTetR = z[..., 2]
        pTetR = z[..., 3]
        mcI = z[..., 4]
        pcI = z[..., 5]

        a = self.params[0]
        b = self.params[1]

        n = self.n
        a0 = self.a0

        mLacI_dot = -mLacI + a/(1 + pcI**n) + a0
        pLacI_dot = -b * (pLacI - mLacI) 
        
        mTetR_dot = -mTetR + a/(1 + pLacI**n) + a0
        pTetR_dot = -b * (pTetR - mTetR) 
                
        mcI_dot = -mcI + a/(1 + pTetR**n) + a0
        pcI_dot = -b * (pcI - mcI) 
        
        zdot = torch.cat([mLacI_dot.unsqueeze(-1),
                            pLacI_dot.unsqueeze(-1),
                            mTetR_dot.unsqueeze(-1),
                            pTetR_dot.unsqueeze(-1),
                            mcI_dot.unsqueeze(-1),
                            pcI_dot.unsqueeze(-1)], dim=-1)
        
        return zdot

    def get_fixed_pts_org(self):
        pass

    def get_dist_from_bifur_helper(self, verbose=False):
        n = self.n
        a0 = self.a0
        a = self.params[0]
        b = self.params[1]

        def equation(x, a, a0, n):
            return a / (1 + x**n) + a0 - x
        
        # Initial guess for x
        p0 = 1.0

        # Solve the equation using fsolve
        p = fsolve(equation, p0, args=(a, a0, n))

        if verbose:
            print("Solution for p:", p)

        xi = - (a * n * p**(n - 1)) / (1 + p**n)**2

        if verbose:
            print("Solution for xi:", xi)

        # check condition for stability
        return ((b + 1)**2 / b) - 3 * xi**2 / (4 + 2 * xi)
    
    def get_topology(self, verbose=False):
        diff = self.get_dist_from_bifur_helper(verbose=verbose)
        # check condition for stability
        if diff > 0:
            topo = [topo_attr_spiral]
        else:
            topo = [topo_period_attr, topo_repeller] # don't know that actually has a period...
        return topo

    
    def get_dist_from_bifur(self, verbose=False):
        return abs(self.get_dist_from_bifur_helper(verbose=verbose)) # this is a proxy!!

    @staticmethod
    def get_bifurcation_curve(n_points=100):
        n = Repressilator.n
        a0 = Repressilator.a0

        a_min = Repressilator.recommended_param_ranges[0][0]; a_max = Repressilator.recommended_param_ranges[0][1]
        a = np.linspace(a_min, a_max, n_points)
        
        def equation(x, a, a0, n):
            return a / (1 + x**n) + a0 - x

        # Initial guess for x
        p0 = np.full(n_points, 1.0)

        # Solve the equation using fsolve
        p = fsolve(equation, p0, args=(a, a0, n))

        A = -a*n*p**(n-1) / (1 + p**n)**2
        beta1 = (3*A**2 - 4*A - 8) / (4*A+8) + (A*np.sqrt(9*A**2 - 24*A-48)) / (4*A+8)
        beta2 = (3*A**2 - 4*A - 8) / (4*A+8) - (A*np.sqrt(9*A**2 - 24*A-48)) / (4*A+8)
        df_lower = pd.DataFrame({r'$a$': a, r'$b$': beta1})
        df_upper = pd.DataFrame({r'$a$': a, r'$b$': beta2})
        b_min = Repressilator.recommended_param_ranges[1][0]; b_max = Repressilator.recommended_param_ranges[1][1]
        df_lower = df_lower[(df_lower[r'$b$'] > b_min) & (df_lower[r'$b$'] < b_max)]
        df_upper = df_upper[(df_upper[r'$b$'] > b_min) & (df_upper[r'$b$'] < b_max)]
        return [df_lower, df_upper]