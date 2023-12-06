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
    """
    Transform cartesian coordinates to polar coordinates
    """
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return r, theta

def polar_derivative_to_cartesian_derivative(r, theta, rdot, thetadot):
    """
    Transform polar derivatives to cartesian derivatives
    """
    xdot = torch.cos(theta) * rdot - r * torch.sin(theta) * thetadot
    ydot = torch.sin(theta) * rdot + r * torch.cos(theta) * thetadot
    return xdot, ydot

########################################################## classical systems  ######################################################################

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


class SubcriticalHopf(FlowSystemODE):
    """
    Subcritical Hopf bifurcation:
        rdot = mu * r + r^3 - r^5
        thetadot = omega + b*r^2
    where:
    - mu controls stability of fixed point at the origin
    - omega controls frequency of oscillations
    - b controls dependence of frequency on amplitude

    Strogatz, p.252
    """

    min_dims = [-1, -1]
    max_dims = [1, 1]
    recommended_param_ranges=[[-0.5,0.25],[-1.,1.],[-1.,1.]]
    recommended_param_groups=[recommended_param_ranges]
    
    short_name='subhopf'
    bifurp_desc = r'$\mu$'
    param_descs = [r'$\mu$', r'$\omega$', r'$b$']


    def forward(self, t, z, **kwargs):
        x = z[..., 0]
        y = z[..., 1]

        mu = self.params[0]
        omega = self.params[1]
        b = self.params[2]

        r,theta = cartesian_to_polar(x, y)
        
        rdot = mu * r + r**3 - r**5
        thetadot = omega + b*r**2
        
        xdot, ydot = polar_derivative_to_cartesian_derivative(r, theta, rdot, thetadot)
        
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot

    def get_fixed_pts_org(self):
        return [(0.,0.)]
    
    def get_topology(self):
        
        mu = self.params[0].numpy()
        if mu < -1/4:
            topos = [topo_attr_spiral]
        elif mu > -1/4 and mu < 0:
            topos = [topo_attr_spiral, topo_period_rep, topo_period_attr]
        elif mu > 0:
            topos = [topo_rep_spiral, topo_period_attr]
        else:
            topos = [topo_degenerate]
            
        return topos
    
    def get_dist_from_bifur(self):
        mu = self.params[0].numpy()
        return np.minimum(np.linalg.norm(mu), np.linalg.norm(mu + 1/4))


    @staticmethod
    def get_bifurcation_curve(n_points=100):
        mu1 = np.full(n_points, 0)
        mu2 = np.full(n_points, -1/4)
        omega_min = SupercriticalHopf.recommended_param_ranges[1][0]; omega_max = SupercriticalHopf.recommended_param_ranges[1][1]
        omega = np.linspace(omega_min, omega_max, n_points)
        b_min = SupercriticalHopf.recommended_param_ranges[2][0]; b_max = SupercriticalHopf.recommended_param_ranges[2][1]
        b = np.linspace(b_min, b_max, n_points)
        return [pd.DataFrame({r'$\mu$': mu1, 
                             r'$\omega$': omega, 
                             r'$b$': b}), 
                pd.DataFrame({r'$\mu$': mu2, 
                             r'$\omega$': omega, 
                             r'$b$': b})]
    

########################################################## more complex systems ######################################################################


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
    max_dims = [10,20] 
    
    recommended_param_ranges=[[3,19], [2, 6]]
    recommended_param_groups=[recommended_param_ranges]
    
    short_name='bz'
    bifurp_desc = r'$(a, \frac{3a}{5} - \frac{25}{a})$'
    param_descs = [r'$a$', r'$b$']


    def forward(self, t, z, **kwargs):
        x = z[..., 0] 
        y = z[..., 1] 

        a,b = self.params
        
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
        
        return np.array([[-1 + 4*y*(x**2-1) / (1 + x**2)**2, -4*x / (1 + x**2)], 
                         [b*y*(x**2 - 1) / (1 + x**2)**2 + b, -b * x / (1 + x**2)]])

    
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
    recommended_param_ranges=[[.01,.11],[.02,1.2]] # cutting off param regime..

    recommended_param_groups = [recommended_param_ranges]
    short_name='sl'
    bifurp_desc = r'$(a,\sqrt{\frac{1}{2} (1-2a \pm \sqrt{1-8a})})$'
    param_descs = [r'$a$', r'$b$']

        
    def forward(self, t, z, **kwargs):
        x = z[...,0]
        y = z[...,1]
        
        a, b = self.params
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
    """
    min_dims = [-3., -3.]
    max_dims = [3., 3.]
    
    recommended_param_ranges = [[-1, 1]]
    recommended_param_groups=[recommended_param_ranges]

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

    recommended_param_ranges=[[0.0,1.0], [1.0,1.0], [-1.0,1.0], [1.0,1.0]]
    recommended_param_groups = [recommended_param_ranges]
    param_descs = [r'$a$', r'$b$', r'$c$', r'$d$']
    bifurp_desc = r'$c'
    plot_param_idx = [2,0]
    min_dims = [-4.2, -4.2]
    max_dims = [4.2, 4.2]

    def __init__(self, params, **kwargs):
        super().__init__(params=params, **kwargs)
        a,b,c,d = self.params

        self.g = lambda x: a * x + b * x**3
        self.f = lambda x: c + d * x**2
    
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
    recommended_param_ranges=[[1.0,2.0], [-1.0,1.0], [1.0,1.0]] 
    recommended_param_groups = [recommended_param_ranges]

    param_descs = [r'$a$', r'$b$', r'$c$'] 
    bifurp_desc = r'$b$'
    plot_param_idx = [1,0]

    min_dims = [-1.5, -1.5]
    max_dims = [1.5, 1.5]
    

    def __init__(self, params, **kwargs):
        super().__init__(params=params, **kwargs)
        assert(self.params[0] > 0)
        self.g = lambda x: 1 / (1 + torch.exp(-self.params[0] * x)) - 0.5
        assert(self.params[2] > 0)
        self.f = lambda x: self.params[1] + self.params[2] * x**2
        
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
    a0 = 0.2 
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
            topo = [topo_period_attr, topo_repeller] 
        return topo

    
    def get_dist_from_bifur(self, verbose=False):
        return abs(self.get_dist_from_bifur_helper(verbose=verbose)) 

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

