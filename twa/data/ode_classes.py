from .ode import FlowSystemODE
import pandas as pd
import torch
import os
import numpy as np
import torch.nn.functional as F
from twa.data.polynomials import sindy_library
import matplotlib.pyplot as plt

from .topology import *

eps = 1e-6

############################################################################## systems with hopf bifurcation ##############################################################################
def cartesian_to_polar(x, y):
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return r, theta

def polar_derivative_to_cartesian_derivative(r, theta, rdot, thetadot):
    xdot = torch.cos(theta) * rdot - r * torch.sin(theta) * thetadot
    ydot = torch.sin(theta) * rdot + r * torch.cos(theta) * thetadot
    return xdot, ydot
        
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

class BZreaction(FlowSystemODE):
    """
    BZ reaction (undergoing Hopf bifurcation):
        xdot = a - x - 4*x*y / (1 + x^2)
        ydot = b * x * (1 - y / (1 + x^2))
    where:
        - a, b depend on empirical rate constants and on concentrations of slow reactants 

    Strogatz, p.256
    """
    
    min_dims = [-1,-1] 
    max_dims = [1,1] 
    recommended_param_ranges=[[2,19], [2, 6]]
    recommended_param_groups=[recommended_param_ranges]
    
    short_name='bz'
    bifurp_desc = r'$(a, \frac{3a}{5} - \frac{25}{a})$'
    param_descs = [r'$a$', r'$b$']

    # eq_string = r'$\dot{x}_0 = x_1; \dot{x}_1=%.02f x_1 + x_0 - x_0^2 + x_0x_1$'

    def __init__(self, params,  min_dims=None, max_dims=None, foldx=5, foldy=10, **kwargs):
        min_dims = self.min_dims if min_dims is None else min_dims
        max_dims = self.max_dims if max_dims is None else max_dims
        super().__init__(params=params, min_dims=min_dims, max_dims=max_dims, **kwargs)
        self.foldx = foldx
        self.foldy = foldy
        self.shiftx = min(0, self.min_dims[0])
        self.shifty = min(0, self.min_dims[1])
    

    def forward(self, t, z, **kwargs):
        x = z[..., 0] 
        y = z[..., 1] 

        x = self.foldx * (x - self.shiftx)
        y = self.foldy * (y - self.shifty)

        a = self.params[0]
        b = self.params[1]
        xdot = a - x - 4*x*y / (1 + x**2)
        ydot = b * x * (1 - y / (1 + x**2))
                
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        
        return zdot

    def get_fixed_pts_org(self):
        a = self.params[0].numpy()
        b = self.params[1].numpy()
        x_st = a / 5
        y_st = 1 + (a/5)**2
        
        return [(x_st, y_st)]
    
    def get_topology(self):
        a = self.params[0].numpy()
        b = self.params[1].numpy()

        b_critical = (3*a/5 - 25/a)
        if b < b_critical:
            topos = [topo_rep_spiral, topo_period_attr]
        elif b > b_critical:
            topos = [topo_attr_spiral]
        else:
            topos = [topo_degenerate]
        
        # checking by fixed point type
        x_st, y_st = self.get_fixed_pts_org()[0]
        J = lambda x,y: 1 / (1 + x**2) * np.array([[3*x**2  - 5, -4*x],[2*b*x**2, -b*x]])
        J_st = J(x_st, y_st)
        topo = get_topology_Jacobian(J_st)
        # assert(topo == topos[0])
        # print(f'Fixed point at ({x_st,y_st}) is {topo_to_str([topo])}')

        return topos
    
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
    
    min_dims = [-1.,-1.]
    max_dims = [1,1]
    
    recommended_param_ranges=[[.01,.11],[.02,1.2]]
    recommended_param_groups = [recommended_param_ranges]
    # eq_string=r'$\dot{x}_0 = -x_0 + {0:%.02f} x_1 + x_0^2 x_1; \dot{x}_1= {1:%.02f} - {0:%.02f} x_1 - x_0x_1$'
    short_name='sl'
    bifurp_desc = r'$(a,\sqrt{\frac{1}{2} (1-2a \pm \sqrt{1-8a})})$'
    param_descs = [r'$a$', r'$b$']

    def __init__(self, params,  min_dims=None, max_dims=None, foldx=1.5, foldy=1.5, **kwargs): # TODO: changed! foldx=1.5, foldy=1.5
        min_dims = self.min_dims if min_dims is None else min_dims
        max_dims = self.max_dims if max_dims is None else max_dims
        super().__init__(params=params, min_dims=min_dims, max_dims=max_dims, **kwargs)
        self.shiftx = min(0, self.min_dims[0])
        self.shifty = min(0, self.min_dims[1])
        self.foldx = foldx
        self.foldy = foldy

    def forward(self, t, z, **kwargs):
        x = z[...,0]
        y = z[...,1]
        v = self.foldx * (x - self.shiftx)
        w = self.foldy * (y - self.shifty)

        a, b = self.params
        # v = (x + 1) * 1.5
        # w = (y + 1) * 1.5
        xdot = -v + a*w + v**2*w
        ydot = b - a*w - v**2*w
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot

    def get_polynomial_representation(self):
        dx = pd.DataFrame(0.0, index=['selkov'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['selkov'], columns=self.polynomial_terms)
        params = [p.numpy() for p in self.params]
        # TODO: fix!
        dx.loc['selkov']['1'] = 1.5 * (params[0]) + 1.875
        dx.loc['selkov']['$x_0$'] = 5.25
        dx.loc['selkov']['$x_1$'] = 1.5 * params[0] + 3.375
        dx.loc['selkov']['$x_0^2$'] = 3.375
        dx.loc['selkov']['$x_0x_1$'] = 6.75
        dx.loc['selkov']['$x_0^2x_1$'] = 3.375
        dy.loc['selkov']['1'] = -1.5 * params[0] + params[1] - 3.375
        dy.loc['selkov']['$x_0$'] = -6.75
        dy.loc['selkov']['$x_1$'] = -1.5 * params[0] - 3.375
        dy.loc['selkov']['$x_0^2$'] = -3.375
        dy.loc['selkov']['$x_0x_1$'] = -6.75 
        dy.loc['selkov']['$x_0^2x_1$'] = -3.375

        return dx, dy

    def get_fixed_pts_org(self):
        a = self.params[0].numpy()
        b = self.params[1].numpy()
        x_st = b
        y_st = b / (a + b**2)
        return [(x_st, y_st)]
    
    def get_topology(self):
        a, b = self.params.numpy()
        root_plus = np.sqrt(1/2 * (1-2*a + np.sqrt(1-8*a)))
        root_minus = np.sqrt(1/2 * (1-2*a - np.sqrt(1-8*a)))

        if root_minus < b and b < root_plus: # stable limit cycle
            topos = [topo_rep_spiral, topo_period_attr]
        elif root_minus > b or b > root_plus: # stable fixed point
            topos = [topo_attr_spiral]
        else:
            topos = [topo_dont_know]

        # fixed pt
        x_st, y_st = self.get_fixed_pts_org()[0]
        J = lambda x,y: np.array([[-1 + 2*x*y, a + x**2], [-2*x*y, -a - x**2]])
        J_st = J(x_st, y_st)
    
        topo = get_topology_Jacobian(J_st)
        if topos[0] == topo_attr_spiral:
            # assert( topo == topo_attr_spiral or topo == topo_attractor)
            pass
        elif topos[0] == topo_rep_spiral:
            pass
            # assert( topo == topo_rep_spiral)
        
        # print(f'Fixed point at ({x_st,y_st}) is {topo_to_str([topo])}')

        return topos
    

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


########################################################### systems oscillating ##########################################################################

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
                    
    eq_string = r'$\dot{x} = r(%.02f - r^2); \dot{\theta} = %.02f$'
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
        x = self.foldx * z[...,0]
        y = self.foldy * z[...,1]
        
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

    recommended_param_ranges=[[0.0,1.0], [0.0,1.0], [-1.0,1.0], [0.0,1.0]] 
    # I want to test both c changing from negative to positive and a changing from positive to negative
    recommended_param_groups = [recommended_param_ranges]
    param_descs = [r'$a$', r'$b$', r'$c$', r'$d$']
    bifurp_desc = r'$c'
    plot_param_idx = [2,0]

    def __init__(self, params, **kwargs):
        super().__init__(params=params, **kwargs)
        a,b,c,d = self.params
        # assert(self.params[[0,1]].sum() > 0)
        self.g = lambda x: a * x + b * x**3

        # assert(c < 0 and d > 0) # allowing negative c
        self.f = lambda x: c + d * x**2
        self.foldx = 4.2 ###8 # 
        self.foldy = 4.2 ###8 # 
    
    def get_fixed_pts_org(self):
        return [(0.,0.)]

    def get_topology(self):

        a,b,c,d = self.params.numpy()
        f_der = lambda x: 2 * d * x
        g_der = lambda x: a + 3 * b * x**2

        topos = []
        J_func = lambda x,y: np.array([[0, 1], [-g_der(x) -f_der(x)*y, -self.f(x)]])
        pts = self.get_fixed_pts_org() #[[0,0]]
        if (a>0 and b<0) or (a<0 and b>0):
            pts.append([np.sqrt(-a/b), 0])
            pts.append([-np.sqrt(-a/b), 0])
        
        for x_st, y_st in pts:
            J = J_func(x_st, y_st)
            topo = get_topology_Jacobian(J)
            if topo == topo_rep_spiral or topo == topo_repeller:
                topos.append(topo_period_attr)
            topos.append(topo)
        
        return topos
    
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



class LienardSigmoid(Lienard):
    """
    Lienard oscillator with polynomial $f$ and sigmoid $g$:
        f(x) = b + c x^2
        g(x) = 1 / (1 + e^(-ax)) - 0.5
    where b < 0 and a,c > 0 there is a limit cycle according to Lienard equations. 
    Here we let b be positive to allow for a fixed point. 
    """

    recommended_param_ranges=[[0.0,1.0], [-1.0,1.0], [0.0,1.0]] # a,b,c
    recommended_param_groups = [recommended_param_ranges]

    param_descs = [r'$a$', r'$b$', r'$c$'] 
    bifurp_desc = r'$b$'
    plot_param_idx = [1,0]
    def __init__(self, params, **kwargs):
        super().__init__(params=params, **kwargs)
        assert(self.params[0] > 0)
        self.g = lambda x: 1 / (1 + torch.exp(-self.params[0] * x)) - 0.5
        # self.g = lambda x: self.params[0] * torch.sin(x)

        # assert(self.params[1] < 0 and self.params[2] > 0)
        assert(self.params[2] > 0)
        self.f = lambda x: self.params[1] + self.params[2] * x**2
        self.foldx = 2.5 ### 4 # 
        self.foldy = 2.5 ### 4 # 
        
    def get_fixed_pts_org(self):
        return [(0.,0.)]

    def get_topology(self):

        a,b,c = self.params.numpy()
        f_der = lambda x: 2 * c * x
        g_der = lambda x: a * np.exp(-a * x) / (1 + np.exp(-a * x))**2

        topos = []
        J_func = lambda x,y: np.array([[0, 1], [-g_der(x) -f_der(x)*y, -self.f(x)]])
        pts = self.get_fixed_pts_org() #[[0,0]]

        # if (a>0 and b<0) or (a<0 and b>0):
        #     pts.append([np.sqrt(-a/b), 0])
        #     pts.append([-np.sqrt(-a/b), 0])
        
        for x_st, y_st in pts:
            J = J_func(x_st, y_st)
            topo = get_topology_Jacobian(J)
            if topo == topo_rep_spiral or topo == topo_repeller:
                topos.append(topo_period_attr)
            topos.append(topo)
        
        return topos
    
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
        labels = ['x', 'y'] if labels is None else labels
        min_dims = [-1.0,-1.0] if min_dims is None else min_dims
        max_dims = [1.0,1.0] if max_dims is None else max_dims
        super().__init__(params=params, labels=labels, min_dims=min_dims, max_dims=max_dims, **kwargs)

        self.poly_order = int(poly_order)
        self.include_sine = include_sine
        self.include_exp = include_exp
        L = self.generate_mesh()
        self.library, self.library_terms = sindy_library(L.reshape(self.num_lattice**self.dim, self.dim), self.poly_order,
                                      include_sine=self.include_sine, include_exp=self.include_exp)
        self.library = self.library.float()
        
    def forward(self, t, z, **kwargs):
        params = self.params.reshape(-1, self.dim).to(self.device) # Error prone
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


    
