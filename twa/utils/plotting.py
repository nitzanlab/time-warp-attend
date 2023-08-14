# plotting functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.colors as colors
from matplotlib import ticker

titlesize = 35
labelsize = 30
ticksize = 25
legendsize = 28


def plot_tradeoff(L, xcol='pc', ycol='l1', xlabel='Sampling probability', ylabel='Smoothed reconstruction error', 
                  color_mean='navy', color_std='royalblue', color_min=None, plot_std=2, xcol_twin=None, twin_values=None,
                  ax=None, pc_opt=None, title=None, label='', groupby=None, 
                  labelsize=labelsize, ticksize=ticksize, titlesize=titlesize, verbose=False,
                  add_fit=False, add_R=False, model_type='huber', **kwargs):
    """
    Plot reconstruction error as a function of sampling probability (alternatively, plot results of any two parameters)
    :param L: dataframe with sampling parameters and errors
    :param pc_opt: add optimal pc to plot
    :param xcol: column name for x axis
    :param ycol: column name for y axis
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param color_mean: color for mean line
    :param color_std: color for standard deviation
    :param color_min: color for minimum error
    :param plot_std: number of standard deviations to plot
    :param ax: axis to plot on
    :param title: title for plot
    :param label: label for legend
    :param groupby: groupby to compute stats
    :param kwargs: additional arguments to pass to plt.plot"""


    ax = plt.subplots(figsize=(6, 6))[1] if ax is None else ax

    groupby = xcol if groupby is None else groupby
    L_grp = L.groupby(groupby)
    s_y = L_grp[ycol].std().values
    y = L_grp[ycol].mean().values
    x = L_grp[xcol].mean().values

    if 'linewidth' not in kwargs.keys():
        kwargs['linewidth'] = 3
    
    ax.plot(x, y, color=color_mean, label=label, **kwargs)
    for i in range(plot_std):
        v = (i + 1)
        ax.fill_between(x, np.array(y) + v * np.array(s_y), y, color=color_std, alpha=0.3/v)
        ax.fill_between(x, np.array(y) - v * np.array(s_y), y, color=color_std, alpha=0.3/v)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    title = title if title is not None else f'{ylabel} vs {xlabel}'
    ax.set_title(title, fontsize=titlesize)

    if pc_opt is not None:
        color_min = color_mean if color_min is None else color_min
        ax.axvline(x=pc_opt, color=color_min, linewidth=3, linestyle='--')

    ax.tick_params(axis='y', labelsize=ticksize)
    ax.tick_params(axis='x', labelsize=ticksize)
    ax.xaxis.label.set_size(labelsize)
    ax.yaxis.label.set_size(labelsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    if xcol_twin is not None and xcol_twin in L.columns:
        ax_twin = ax.twiny()
        # twin_data = L_grp[[xcol, xcol_twin]].mean()
        x2 = L_grp[xcol_twin].mean().values
        n = len(x2)
        n_ticks = 4

        min_xcol_twin = x2.min()
        max_xcol_twin = x2.max()
        x2_values = [t for t in twin_values if (t > min_xcol_twin and t < max_xcol_twin)]
            
        if x2_values is not None and len(x2_values) > 0:
            pts = pd.Series(x, index=x2)
            new_pts = pd.Series(np.nan, index=x2_values)
            x2_x = pd.concat((pts, new_pts))
            x2_x = x2_x.interpolate(method='index').groupby(level=0).mean()
            new_tick_locations = x2_x.loc[x2_values].values
            new_tick_values = x2_values
        else:
            idx = [int(n*i/(n_ticks+1)) for i in np.arange(1, n_ticks)]
            new_tick_locations = x[idx]
            new_tick_values = x2[idx].astype(int)

        ax_twin.set_xlim(ax.get_xlim())
        ax_twin.set_xticks(new_tick_locations)
        ax_twin.set_xticklabels(new_tick_values) # TODO: temp rounding
        ax_twin.tick_params(axis='x', labelsize=ticksize)
        ax_twin.xaxis.label.set_size(labelsize)
    
        # ax_twin.set_xlabel(r"Modified x-axis: $1/(1+X)$")






def plot_diverge_scale(x, y, c, xlabel='', ylabel='', colorlabel='', title='', ax=None, add_colorbar=True, fontsize=20, labelsize=15, s=20, nticks=3):
    if ax is None:
        # fig = plt.figure()
        fig, ax = plt.subplots(constrained_layout=True, tight_layout=False)
    plt.set_cmap('coolwarm')
    vmax = np.abs(c).max()
    vmin = -vmax
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    pl = ax.scatter(x, y, c=c, norm=norm, s=s) #colormap='bwr'
    
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(labelsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(labelsize)
    if add_colorbar:
        cbar = plt.colorbar(pl, ax=ax)
        cbar.ax.set_ylabel(colorlabel, rotation=270, labelpad=30)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.ax.tick_params(labelsize=labelsize)
        cbar.locator = tick_locator
        cbar.update_ticks()
    ax.set_title(title, fontsize=fontsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nticks))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nticks))
        
    # ax.axis('off')




# x = sysp[:,0]
# y = sysp[:,1]
# z = exp_output[sys][:,1]

def get_prediction(x,y,z, npts=40, p=0.5):
    A_func = lambda x,y: np.array([x*0+1, x, y, x**2, x**2*y, x**2*y**2, y**2, x*y**2, x*y]).T
    A = A_func(x,y)

    p = 0.5
    # solve least error with variable norm, p
    coeff_lstq, r, rank, s = np.linalg.lstsq(A, z)
    sol = minimize(lambda c: np.linalg.norm(A @ c - z, ord=p) + np.linalg.norm(c, ord=2), coeff_lstq)
    # 
    coeff = sol.x

    x_grid = np.linspace(x.min(), x.max(), npts)
    y_grid = np.linspace(y.min(), y.max(), npts)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid, copy=False)
    xy_grid = np.stack((X_grid, Y_grid), axis=-1).reshape((-1,2))
    A_grid = A_func(xy_grid[:,0],xy_grid[:,1])
    z_grid = A_grid @ coeff

    return xy_grid, z_grid
    # plt.scatter(xy_grid[:,0], xy_grid[:,1], c=z_grid, s=50)
