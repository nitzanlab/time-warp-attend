import os
import glob
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
from twa.utils import ensure_dir, write_yaml, read_yaml, glob_re, plot_tradeoff
from twa.data import FlowSystemODE, topo_point_vs_cycle, pt_attr_idx, cyc_attr_idx, SystemFamily
from sklearn.decomposition import PCA
from .utils import find_optimal_cutoff
from sklearn.metrics import silhouette_score, roc_curve, auc
from twa.utils import plot_diverge_scale, get_prediction
import scanpy as sc
import scvelo as scv

from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr

torch.manual_seed(0)


class Evaluator:
    def __init__(self, exp_descs, sys_descs, sys_gt_descs=None, ode_syss=None, num_lattice=64, tt='test', outdir='../output/', data_dir=None, verbose=False, repeats=np.inf):
        """
        Initialize evaluation class.
        exp_descs: dict of experiment descriptions
        sys_descs: dict of system descriptions
        sys_gt_descs: dict of the relevant dirname for each system labels
        ode_syss: dict of the ODE system name for each system
        """

        assert(isinstance(exp_descs, dict))
        self.exp_descs = exp_descs
        self.sys_descs = sys_descs 
        self.sys_gt_descs = sys_gt_descs if sys_gt_descs is not None else {k: k for k in self.sys_descs.keys()}
        self.ode_syss = ode_syss if ode_syss is not None else {k: k for k in self.sys_descs.keys()}
        self.exps = list(exp_descs.keys())
        self.nexps = len(self.exps)

        self.outdir = outdir
        self.data_dir = os.path.join(outdir, 'data') if data_dir is None else data_dir
        self.result_dir = os.path.join(outdir, 'results')
        if (not os.path.isdir(self.outdir)) or (not os.path.isdir(self.data_dir)) or (not os.path.isdir(self.result_dir)):
            raise ValueError('Invalid output directory.')
        
        self.repeats = repeats
        self.num_lattice = num_lattice
        self.tt = tt
        
        # verify all ground truth syss exist
        for sys, sys_gt in self.sys_gt_descs.items():
            ddir = os.path.join(self.data_dir, sys_gt)
            if not os.path.isdir(ddir):
                print(f'Ground truth for {sys} does not exist ({sys_gt} is missing).')
                self.sys_descs.pop(sys)
                self.sys_gt_descs.pop(sys)

        self.syss = list(self.sys_descs.keys())
        self.nsyss = len(self.syss)

        # Read ground truth and results
        if verbose:
            print('Reading ground truth...')

        self.topos = {}
        self.sysps = {}
        self.dists = {}
        self.y_pts = {}
        self.bifurps = {}
        self.fixed_pts = {}

        for sys, sys_gt in self.sys_gt_descs.items():
            if verbose:
                print(sys)
            ddir = os.path.join(self.data_dir, sys_gt)
            self.topos[sys] = np.load(os.path.join(ddir, f'topo_{tt}.npy'))
            self.sysps[sys] = np.load(os.path.join(ddir, f'sysp_{tt}.npy'))
            fname = os.path.join(ddir, f'dists_{tt}.npy')
            self.dists[sys] = None
            if os.path.isfile(fname):
                try:
                    self.dists[sys] = np.load(fname).flatten()
                except:
                    print('Error loading dists for ', sys)

            dist = self.dists[sys]
            dist = dist.reshape(-1) if dist is not None else None
            topo = self.topos[sys]
            
            label = topo_point_vs_cycle(topo)

            # assuming binary topology classification
            has_cyc = label['Periodic'].values
            self.y_pts[sys] = 1 - has_cyc
            self.bifurps[sys] = (-dist * (1 - has_cyc) + dist * has_cyc) if dist is not None else None

            fname = os.path.join(ddir, f'fixed_pts_{tt}.npy')
            if os.path.isfile(fname):
                fixed_pt = np.load(fname)
                self.fixed_pts[sys] = fixed_pt

        self.adata = None
        if 'pancreas_clusters_random_bin' in self.syss:
            self.adata = sc.read_h5ad(os.path.join(self.data_dir,'pancreas', 'adata.h5'))

        if verbose:
            print('Reading predictions...')

        self.exp_outputs = {} # logits
        self.exp_pred_pts = {} # point logit positive
        self.exp_embeddings = {} # embeddings
        self.exp_repeats = {} # list of repeats for each experiment

        for exp in self.exps:
            # for each experiment, get results with all folders (self.repeats)
            exp_base_name = os.path.join(self.result_dir, exp)
            exp_repeat_fnames = glob.glob(exp_base_name + '_*')
            exp_repeat_fnames = glob_re(fr'{exp_base_name}_+[0-9]+$', exp_repeat_fnames)
            if len(exp_repeat_fnames) > self.repeats:
                exp_repeat_fnames = exp_repeat_fnames[:self.repeats]
            
            
            if verbose:
                print(exp)
            exp_reps = []
            for exp_rep_fname in exp_repeat_fnames:
                if verbose:
                    print(exp_rep_fname)
            
                exp_rep = os.path.basename(exp_rep_fname)
                exp_reps.append(exp_rep)

                embeddings = {}
                outputs = {}
                pred_pts = {}
                for sys in self.syss:
                    fname = os.path.join(exp_rep_fname, sys, f'embeddings_{tt}.npy')
                    if os.path.isfile(fname):
                        embeddings[sys] = np.load(fname)
                    fname = os.path.join(exp_rep_fname, sys, f'outputs_{tt}.npy')
                    if os.path.isfile(fname):
                        outputs[sys] = np.load(fname)
                        pred_pts[sys] = outputs[sys][:, pt_attr_idx] > 0

                self.exp_embeddings[exp_rep] = embeddings
                self.exp_outputs[exp_rep] = outputs
                self.exp_pred_pts[exp_rep] = pred_pts
            self.exp_repeats[exp] = exp_reps

            self.res = None

    
    def gather_stats(self):
        """
        Compute sensitivity, specificity, accuracy for point class. 
        """
        if self.res is not None:
            return self.res
        res = []
        # considering 0-true as point attractor, else, cycle
        for exp_rep in self.exp_pred_pts.keys():

            pred_pts = self.exp_pred_pts[exp_rep]
            outputs = self.exp_outputs[exp_rep]

            for sys in pred_pts.keys():
                y_pt = self.y_pts[sys]
                idx_pt = np.where(y_pt)[0]
                idx_cyc = np.where(~y_pt)[0]
                repeat = exp_rep.split('_')[-1]
                exp = '_'.join(exp_rep.split('_')[:-1])
                
                pred_pt = pred_pts[sys]
                output = outputs[sys]

                auc_val = None
                try:
                    fpr, tpr, thresholds = roc_curve(y_pt, output[:, pt_attr_idx])
                    auc_val = auc(fpr, tpr)
                except:
                    pass
                
                res.append( {'exp': exp, 'repeat':repeat, 'system': sys,
                            'Sens': np.round(np.mean(pred_pt[idx_pt]), 2), 
                            'Spec': np.round(np.mean(1-pred_pt[idx_cyc]), 2),
                            'Accu': np.round(np.mean(pred_pt==y_pt), 2),
                            'AUC': auc_val} )
                
        self.res = pd.DataFrame(res)
        
        return self.res
    

    def get_stats(self, syss=None, stats=['Accu', 'AUC']):
        """
        Get statistics, e.g. accuracy ('Accu'), sensitivity ('Sens') and specificity ('Spec') results.
        """
        syss = self.syss if syss is None else syss
        res = self.gather_stats()
        df = pd.melt(res, id_vars=['exp', 'system'], value_vars=stats)
        df = df.groupby(['exp', 'system', 'variable']).mean().pivot_table(index=['exp', 'variable'], columns='system', values='value')
        df.index.names = ['','']
        df.columns.name = ''
        df = df.loc[self.exps].rename(index=self.exp_descs)
        df = df[syss].rename(columns=self.sys_descs)
        df['Average'] = df.mean(axis=1)
        return df
        
        
    def get_stats_std(self, syss=None, stats=['Accu', 'AUC']):
        """
        Get accuracy, sensitivity and specificity results.
        'Sens', 'Spec', 'Accu'
        """
        syss = self.syss if syss is None else syss
        res = self.gather_stats()
        df = pd.melt(res, id_vars=['exp', 'system'], value_vars=stats)
        df = df.groupby(['exp', 'system', 'variable']).std().pivot_table(index=['exp', 'variable'], columns='system', values='value')
        df.index.names = ['','']
        df.columns.name = ''
        df = df.loc[self.exps].rename(index=self.exp_descs)
        df = df[syss].rename(columns=self.sys_descs)
        return df
    

    @staticmethod
    def print_mean_std(df_mean, df_std, r=2, highlight=True):
        """
        Print mean and standard deviation.
        """
        df_mean = df_mean.round(r)
        df_std = df_std.round(r)

        df_mean.index = df_mean.index.get_level_values(0)
        df_std.index = df_std.index.get_level_values(0)
        cols = df_std.columns
        cols_left = list(set(df_mean.columns) - set(cols))
        df = df_mean[cols].astype(str) + u"\u00B1" + df_std[cols].astype(str)
        df[cols_left] = df_mean[cols_left].astype(str)
        df = df.style.format()

        if highlight:
            # loop through rows and find which column for each row has the highest value
            for col in df_mean.columns:
                row = df_mean[col].idxmax()
                # redo formatting for a specific cell
                df = df.format(lambda x: "\\textbf{" + x + "}", subset=(row, col))

        print(df.to_latex())


    def print_stat(self, syss=None, stat=['Accu'], **kwargs):
        """
        Print accuracy results.
        """
        if len(stat) > 1:
            raise ValueError('Only one stat allowed.')
        
        df_mean = self.get_stats(stats=stat, syss=syss)
        df_std = self.get_stats_std(stats=stat, syss=syss)

        Evaluator.print_mean_std(df_mean, df_std, **kwargs)
    
    
    def sample_exp_from_repeats(self):
        """
        Sample one experiment from each repeat.
        """
        single_exp_descs = {}
        for exp in self.exps:
            single_exp_descs[exp] = np.random.choice(self.exp_repeats[exp])
        return single_exp_descs
    
    
    def plot_bifurcation(self, exp, syss=None, s=50, verbose=False, **kwargs_pred):
        """
        Plot confidence and inferred bifurcation diagrams.
        """
        labelsize = 25
        fontsize = 30
        lw = 3
        syss = self.syss if syss is None else syss
        syss = [sys for sys in syss if self.ode_syss[sys] is not None]
        
        if verbose:
            print('Showing results for experiment: ', exp)

    
        ncols = len(syss)
        fig, ax = plt.subplots(1, ncols, figsize=(ncols*5, 5), tight_layout=False, constrained_layout=True)
        
        if ncols == 1:
            ax = [ax]

        fit_value = {}
        for sys in syss:
            fit_value[sys] = np.mean([np.sign(self.exp_outputs[exp_rep][sys]) for exp_rep in self.exp_repeats[exp] if sys in self.exp_outputs[exp_rep].keys()], axis=0)
            
        for isys, sys in enumerate(syss):
            DE = SystemFamily.get_generator(self.ode_syss[sys])
            param_idx1 = DE.plot_param_idx[0]
            param_idx2 = DE.plot_param_idx[1] if len(DE.plot_param_idx) > 1 else param_idx1

            bifurp = self.bifurps[sys]
            sysp = self.sysps[sys]

            y_cyc = fit_value[sys][:, cyc_attr_idx]
            tit = f'{self.sys_descs[sys]}'

            param_descs = DE.param_descs
            
            xlabel=param_descs[param_idx1]
            ylabel=param_descs[param_idx2]
            colorlabel = ''
                
            plot_diverge_scale(sysp[:,param_idx1], sysp[:,param_idx2], y_cyc, xlabel=xlabel, 
                                ylabel=ylabel, colorlabel=colorlabel, title=tit, ax=ax[isys], 
                                add_colorbar=False, labelsize=labelsize, fontsize=fontsize, s=s, vmin=-1, vmax=1)
            
    
            # adding true bifurcation boundaries
            bifur_dfs = DE.get_bifurcation_curve()
            if bifur_dfs is not None:
                for bifur_df in bifur_dfs:
                    ax[isys].plot(bifur_df.values[:,param_idx1], bifur_df.values[:,param_idx2], color='k', lw=lw)

        plt.locator_params(axis='both', nbins=4)


    def plot_dist_from_bifur(self, exp, syss=None, verbose=False):
        """
        Plot confidence distribution as a function of the distance from bifurcation curve.
        """
        syss = self.syss if syss is None else syss
        ncols = len(syss)
        fig, ax = plt.subplots(1,ncols,figsize=(5*ncols,7), sharey=True, constrained_layout=True)
        bins = 10
        xlabel=''
        ylabel=''
        for isys, sys in enumerate(syss):
            sysp = self.sysps[sys]
            bifurp = self.bifurps[sys]
            fit_value = np.mean([np.sign(self.exp_outputs[exp_rep][sys]) for exp_rep in self.exp_repeats[exp] if sys in self.exp_outputs[exp_rep].keys()], axis=0)

            df_cyc = pd.DataFrame({ 'dist': np.abs(bifurp[bifurp > 0]), 'confidence': np.abs(fit_value[bifurp > 0,0])})
            df_pt = pd.DataFrame({ 'dist': np.abs(bifurp[bifurp < 0]), 'confidence': np.abs(fit_value[bifurp < 0,0])})

            # measures of agreement:
            
            # corr_pt = mutual_info_regression(df_pt[['dist']], df_pt[['confidence']])[0]
            # corr_cyc = mutual_info_regression(df_cyc[['dist']], df_cyc[['confidence']])[0]
            
            # corr_pt = np.corrcoef(df_pt['dist'], df_pt['confidence'])[0,1]
            # corr_cyc = np.corrcoef(df_cyc['dist'], df_cyc['confidence'])[0,1]
            
            corr_pt = spearmanr(df_pt['dist'], df_pt['confidence']).correlation
            corr_cyc = spearmanr(df_cyc['dist'], df_cyc['confidence']).correlation

            if verbose:
                print(sys, corr_pt, corr_cyc)
            tit = f'{self.sys_descs[sys]}\n({corr_pt:.2f}, {corr_cyc:.2f})'
            # tit = f'({corr_pt:.2f}, {corr_cyc:.2f})'

            
            df_cyc['dist_bin'] = pd.cut(df_cyc['dist'], bins=bins)
            df_cyc['dist_mid'] = df_cyc['dist_bin'].apply(lambda x: x.mid).astype(float)
            plot_tradeoff(df_cyc, xcol='dist_mid', ycol='confidence', groupby='dist_mid', plot_std=1, color_mean='r', color_std='r', xlabel=xlabel, ylabel=ylabel, ax=ax[isys])

            df_pt['dist_bin'] = pd.cut(df_pt['dist'], bins=bins)
            df_pt['dist_mid'] = df_pt['dist_bin'].apply(lambda x: x.mid).astype(float)
            plot_tradeoff(df_pt, xcol='dist_mid', ycol='confidence', groupby='dist_mid', plot_std=1, color_mean='b', color_std='b', xlabel=xlabel, ylabel=ylabel, ax=ax[isys], title=tit)

        ax[0].set_ylabel('Confidence', fontsize=30)
        fig.supxlabel('Distance from bifurcation', fontsize=30)


    def plot_pancreas(self, exp):
        """
        Plot predicted cell cycle score
        """
        splitby = 'clusters_random_bin'
        sys = f'pancreas_{splitby}'
        nrows = 1
        ncols = 3
        fontsize = 20
        legend_fontsize = 15

        # load 
        self.adata.obs['cc_pred'] = 0
        self.adata.obs['point_pred'] = 0
        # exp_output = self.exp_outputs[exp][sys]
        exp_output = np.mean([np.sign(self.exp_outputs[exp_rep][sys]) for exp_rep in self.exp_repeats[exp]], axis=0)
        samples = list(self.adata.obs[splitby].unique())
        
        for i,t in enumerate(samples):
            self.adata.obs.loc[self.adata.obs[splitby] == t, 'cc_pred'] = exp_output[:,cyc_attr_idx][i]
            self.adata.obs.loc[self.adata.obs[splitby] == t, 'point_pred'] = exp_output[:,pt_attr_idx][i]

        fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows*5))
        scv.pl.velocity_embedding_stream(self.adata, basis='umap', show=False, ax=ax[0])
        ax[0].set_title('Velocity in gene expression space', fontsize=fontsize)

        scv.pl.scatter(self.adata, color=f'frac_by_{splitby}', smooth=True, show=False, ax=ax[1], legend_fontsize=legend_fontsize, cmap='coolwarm')
        ax[1].set_title('True cell-cycle score', fontsize=fontsize)

        scv.pl.scatter(self.adata, color='cc_pred', smooth=True, show=False, ax=ax[2], legend_fontsize=legend_fontsize, cmap='coolwarm')
        ax[2].set_title('Predicted cell-cycle score', fontsize=fontsize)
        print('Corr:', np.corrcoef(self.adata.obs['cc_pred'], self.adata.obs[f'frac_by_{splitby}'])[0,1])
        plt.show()


    def plot_output_distribution(self, syss=None, exps=None, **kwargs):
        """
        Plot output distribution.
        """
        exps = self.exps if exps is None else exps
        syss = self.syss if syss is None else syss

        ncols = len(exps)
        nrows = len(syss)
        fig, ax  = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows))
        ax = ax.reshape((nrows, ncols))
        
        ticksize = 15
        for isys, sys in enumerate(syss):
            y_pt = self.y_pts[sys]
            idx_pt = np.where(y_pt)[0]
            idx_cyc = np.where(~y_pt)[0]
            for iexp, exp in enumerate(exps):
                mean_pts = []
                mean_cycs = []
                for exp_rep in self.exp_repeats[exp]:
                    output = self.exp_outputs[exp_rep][sys]
                    mean_pts.append(output[idx_pt, pt_attr_idx].mean())
                    mean_cycs.append(output[idx_cyc, pt_attr_idx].mean())

                df = pd.DataFrame({'mean pt': mean_pts, 'mean cyc': mean_cycs})
                maxlim = max(np.abs(df['mean pt'].max()), np.abs(df['mean cyc'].max()), np.abs(df['mean pt'].min()), np.abs(df['mean cyc'].min()))
                
                ax[isys,iexp].hist(df['mean pt'], label='point', alpha=0.5, color='blue')
                ax[isys,iexp].hist(df['mean cyc'], label='cycle', alpha=0.5, color='red')
                
                ax[isys,iexp].set_xlim(-maxlim, maxlim)
                ax[isys,iexp].axvline(0, color='black', linestyle='--')
                ax[isys,iexp].xaxis.set_major_locator(plt.MaxNLocator(3))
                ax[isys,iexp].yaxis.set_major_locator(plt.MaxNLocator(3))
                ax[isys,iexp].tick_params(axis='y', labelsize=ticksize)
                ax[isys,iexp].tick_params(axis='x', labelsize=ticksize)
                plt.grid(False)        
                ax[0,iexp].set_title(self.exp_descs[exp], fontsize=20)
            ax[isys,0].set_ylabel(self.sys_descs[sys], fontsize=20)

        xlabel = 'Point logit value'
        fig.supxlabel(xlabel, fontsize=15)
            
        ax[-1,-1].legend(fontsize=15)

