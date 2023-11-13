import os
import glob
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
from twa.utils import ensure_dir, write_yaml, read_yaml, glob_re
from twa.data import FlowSystemODE, topo_point_vs_cycle, pt_attr_idx, cyc_attr_idx, SystemFamily
from sklearn.decomposition import PCA
from .utils import find_optimal_cutoff
from skdim.id import lPCA, DANCo
from sklearn.metrics import silhouette_score, roc_curve, auc
from twa.utils import plot_diverge_scale, get_prediction
import scanpy as sc
import scvelo as scv
torch.manual_seed(0)



class Evaluator:
    def __init__(self, exp_descs, sys_descs, ode_syss=None, num_lattice=64, tt='test', outdir='../output/', data_dir=None, verbose=False, repeats=np.inf):
        """
        Initialize evaluation class.
        """

        assert(isinstance(exp_descs, dict))
        self.exp_descs = exp_descs
        self.sys_descs = sys_descs 
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
        for sys in self.sys_descs.keys():
            ddir = os.path.join(self.data_dir, sys)
            if not os.path.isdir(ddir):
                print(f'Ground truth for {sys} does not exist.')
                self.sys_descs.pop(sys)
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

        for sys in self.syss:
            if verbose:
                print(sys)
            ddir = os.path.join(self.data_dir, sys)
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
        
        
    def gather_stats_noised(self, exp, desc='noise', ref_sys='simple_oscillator_nsfcl'):
        """
        
        """
        res = []
        
        y_pt = self.y_pts[ref_sys]
        for exp_rep in self.exp_repeats[exp]:
            repeat = exp_rep.split('_')[-1]
            fnames = glob.glob(os.path.join(self.result_dir, exp_rep, f'{ref_sys}_{desc}*'))
            for fname in fnames:
                output = np.load(os.path.join(fname, 'outputs_test.npy'))
                pred_pt = output[:, pt_attr_idx] > 0
                desc_val = fname.split(f'{ref_sys}_{desc}')[-1]
                res.append({'repeat': repeat, desc: desc_val, ' Accu': np.round(np.mean(pred_pt==y_pt), 2)})

        df = pd.DataFrame(res)

        df_mean = df.groupby(desc).mean().T
        df_std = df.groupby(desc).std().T

        Evaluator.print_mean_std(df_mean, df_std, r=2, highlight=False)

        return df

    # def get_acc(self, syss=None):
    #     """
    #     Get accuracy results.
    #     """
    #     syss = self.syss if syss is None else syss
    #     res = self.gather_stats()
    #     df = res[['exp', 'system', 'Accu']].groupby(['exp', 'system']).mean().pivot_table(index='exp', columns='system', values='Accu')
    #     df.index.name = ''
    #     df.columns.name = ''
    #     df = df.loc[self.exps].rename(index=self.exp_descs)
    #     df = df[syss].rename(columns=self.sys_descs)
        
    #     return df

    # def get_stats_std(self, syss=None):
    #     """
    #     Get std of accuracy results.
    #     """
    #     syss = self.syss if syss is None else syss
    #     res = self.gather_stats()
    #     df = res[['exp', 'system', 'Accu']].groupby(['exp', 'system']).std().pivot_table(index='exp', columns='system', values='Accu')
    #     df.index.name = ''
    #     df.columns.name = ''
    #     df = df.loc[self.exps].rename(index=self.exp_descs)
    #     df = df[syss].rename(columns=self.sys_descs)
        
    #     return df
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
    
    def get_silhouette(self, n_neighbors=100):
        res = {}
        comp_sil = True
        comp_lpca = False
        comp_danco = False
        # compute embeddings dimensionality and alignment on test data
        for exp, embeddings in self.exp_embeddings.items():
            fit_sys = '_'.join(exp.split('_')[:3]) # TODO: make generic
            train_embedding = embeddings[fit_sys]
            ntrain = len(train_embedding)
            exp_res = {}
            for sys in self.syss:
                exp_sys_res = {}
                embedding = embeddings[sys]
                if comp_lpca:
                    lpca = lPCA().fit_pw(embedding,
                                                n_neighbors=n_neighbors,
                                                n_jobs=1)
                    exp_sys_res['lpca'] = np.mean(lpca.dimension_pw_)
                if comp_danco:
                    danco = DANCo(k=10)
                    exp_sys_res['danco'] = danco.fit(embedding).dimension_
                nsamples = len(embedding)
                X = np.concatenate((train_embedding, embedding))
                y = np.concatenate((np.zeros(ntrain), np.ones(nsamples)))
                if comp_sil:
                    sil = silhouette_score(X,y)
                    sil = np.nan if fit_sys == sys else sil
                    exp_sys_res['sil'] = sil
                    
                exp_res[sys] = exp_sys_res
                # {'sil': sil, 'lpca': np.mean(lpca.dimension_pw_), }# 'danco': danco.fit(embedding).dimension_}
            
            # # add of all systems together
            # X = np.concatenate([embeddings[sys] for sys in selected_syss])
            # y = np.concatenate([np.ones(len(embeddings[sys]))*i for i,sys in enumerate(selected_syss)])
            # sil = silhouette_score(X,y)
            # lpca = skdim.id.lPCA().fit_pw(X, n_neighbors=n_neighbors,
            #                                 n_jobs=1)
            # # danco = skdim.id.DANCo(k=10)
            # exp_res['All'] = {'sil': sil, 'lpca': np.mean(lpca.dimension_pw_), }# 'danco': danco.fit(X).dimension_}
            res[exp] = pd.DataFrame(exp_res)
                
        res = pd.concat(res)
        res.index.rename(['', ''], inplace=True)
        res = res.droplevel(1)

        res['exp rep'] = res.index
        res['exp'] = res['exp rep'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        df = res.groupby('exp').mean()
        df.index.name = ''
        df.columns.name = ''
        df = df.loc[self.exps].rename(index=self.exp_descs)
        df = df[self.syss].rename(columns=self.sys_descs)

        
        # df_s = res.style.format("{:.2f}")

        # # loop through rows and find which column for each row has the highest value
        # for col in res.columns:
        #     row = res[col].idxmin()
        #     # redo formatting for a specific cell
        #     df_s = df_s.format(lambda x: "\\textbf{" + f'{x:.2f}' + "}", subset=(row, col))
        # print(df_s.to_latex(hrules=True))

        return df

    def sample_exp_from_repeats(self):
        """
        Sample one experiment from each repeat.
        """
        single_exp_descs = {}
        for exp in self.exps:
            single_exp_descs[exp] = np.random.choice(self.exp_repeats[exp])
        return single_exp_descs
    
    def plot_latent(self, single_exp_descs, selected_syss=['bzreaction', 'selkov']):
        # fit pca space and plot all systems

        pca = PCA(n_components=2)
        nrows = 1
        ncols = len(single_exp_descs.keys())
        fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5), constrained_layout=True, tight_layout=False)
        fig2, ax2 = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5), constrained_layout=True, tight_layout=False)
        ax = ax.flatten()
        ax2 = ax2.flatten()
        recompute_pca = False
        exp_latent_pcas = {}
        exp_nsamples_syss = {}

        for iexp, (exp, exp_rep) in enumerate(single_exp_descs.items()):
            fit_sys = '_'.join(exp_rep.split('_')[:3])
            embeddings = self.exp_embeddings[exp_rep]
            embedding = embeddings[fit_sys]

            fit_latent_pca = pca.fit_transform(embedding)

            latent_pcas = {}
            nsamples_syss = {}
            # plt.locator_params(nbins=4)
            ax[iexp].scatter(fit_latent_pca[:,0], fit_latent_pca[:,1], color='black', s=50, alpha=0.5, label='Train dataset')
            for isys, sys in enumerate(selected_syss):
                embedding = embeddings[sys]
                bifurp = self.bifurps[sys]

                # recompute pca
                if recompute_pca:
                    pca = PCA(n_components=2)
                    latent_pca = pca.fit_transform(embedding)
                else:
                    latent_pca = pca.transform(embedding)
                    latent_pcas[sys] = latent_pca
                    nsamples_syss[sys] = len(latent_pca)
                
                ax[iexp].scatter(latent_pca[:,0], latent_pca[:,1], s=50, alpha=0.5, label=self.sys_descs[sys])

            exp_latent_pcas[exp_rep] = latent_pcas
            exp_nsamples_syss[exp_rep] = nsamples_syss
                    
            tit = self.exp_descs[exp]
            ax[iexp].set_xscale('symlog')
            ax[iexp].set_yscale('symlog')
            ax[iexp].set_title(tit, fontsize=25)
            ax[iexp].axis('off')

                # # tit = fr'{sys.title()} (corr: {corr:.2f})'
            if not recompute_pca:
                # ax[isys].scatter(fit_latent_pca[:,0], fit_latent_pca[:,1], color='slategray')
                plot_diverge_scale(fit_latent_pca[:,0], fit_latent_pca[:,1], self.bifurps[fit_sys], xlabel='PC1', ylabel='PC2', colorlabel='Distance from bifurcation', title=tit, ax=ax2[iexp], add_colorbar=False, s=50)
            
            ax2[iexp].set_xscale('symlog')
            ax2[iexp].set_yscale('symlog')
            ax2[iexp].set_title(tit, fontsize=25)
            ax2[iexp].axis('off')
                # # ax[iexp].locator_params(nbins=4)
                # ax[iexp].xaxis.set_tick_params(labelsize=fontsize)
                # ax[iexp].yaxis.set_tick_params(labelsize=fontsize)
                # ax[iexp].xaxis.set_major_locator(plt.MaxNLocator(nticks))
                # ax[iexp].yaxis.set_major_locator(plt.MaxNLocator(nticks))
            

        ax[iexp].legend(fontsize=25, markerscale=2, frameon=False)


    def plot_bifurcation(self, exp, syss=None, acc_thr=0.5, ref_sys='simple_oscillator_nsfcl', s=50, verbose=False, **kwargs_pred):
        """
        Plot confidence and inferred bifurcation diagrams.
        """
        syss = self.syss if syss is None else syss
        syss = [sys for sys in syss if self.ode_syss[sys] is not None]
        # here, set const thr
        # thr = 1
        # instead, empirically determine threshold based on low accuracy in train test data
        
        if verbose:
            print('Showing results for experiment: ', exp)
        # TODO: temporary set test sys
         # select threshold where accuracy > 0.5
        y_pt = self.y_pts[ref_sys]
        bins = np.arange(-2,2,0.1)
        # add -inf and inf to bins
        bins = np.concatenate(([-np.inf], bins, [np.inf]))
        output_accs = []
        for exp_rep in self.exp_repeats[exp]:
            output = self.exp_outputs[exp_rep][ref_sys]
            pred_pt = output[:, pt_attr_idx]
            pred_cyc = output[:, cyc_attr_idx]
            # bin accuracy by output values
            acc_pt = (pred_pt>0) == y_pt
            acc_cyc = (pred_cyc>0) == (1-y_pt)
            output_accs.append(pd.DataFrame({'pred_pt': pred_pt, 'acc_pt': acc_pt, 'pred_cyc': pred_cyc, 'acc_cyc': acc_cyc}))

        output_accs_df = pd.concat(output_accs)
        output_accs_df['pred_pt_bin'] = pd.cut(output_accs_df['pred_pt'], bins=bins)
        output_accs_df['pred_cyc_bin'] = pd.cut(output_accs_df['pred_cyc'], bins=bins)
        acc_pt_bin = output_accs_df.groupby('pred_pt_bin').mean()['acc_pt']
        acc_cyc_bin = output_accs_df.groupby('pred_cyc_bin').mean()['acc_cyc']

        if verbose:
            acc_pt_bin.plot(color='b')
            acc_cyc_bin.plot(color='r', linestyle='--')
            plt.axhline(acc_thr, color='k', linestyle=':')
            plt.xlabel('Point attractor confidence', fontsize=20)
            plt.ylabel('Accuracy', fontsize=20)
            plt.title('Point attractor', fontsize=20)
            plt.show()

        # examine bins after 0, for first bin where accuracy > 0.5
        acc_pt_bin_low = acc_pt_bin.index.to_series().apply(lambda x: x.left)
        acc_pt_bin_high = acc_pt_bin.index.to_series().apply(lambda x: x.right)
        acc_pt_bin = pd.DataFrame(acc_pt_bin)
        acc_pt_bin['low'] = acc_pt_bin_low.astype(float)
        acc_pt_bin['high'] = acc_pt_bin_high.astype(float)
        thr_pt = acc_pt_bin[(acc_pt_bin['low']>=0) & (acc_pt_bin['acc_pt']>acc_thr)]['high'].min()

        acc_cyc_bin_low = acc_cyc_bin.index.to_series().apply(lambda x: x.left)
        acc_cyc_bin_high = acc_cyc_bin.index.to_series().apply(lambda x: x.right)
        acc_cyc_bin = pd.DataFrame(acc_cyc_bin)
        acc_cyc_bin['low'] = acc_cyc_bin_low.astype(float)
        acc_cyc_bin['high'] = acc_cyc_bin_high.astype(float)
        thr_cyc = acc_cyc_bin[(acc_cyc_bin['high']<=0) & (acc_cyc_bin['acc_cyc']>acc_thr)]['low'].max()
        

        thr = max(thr_pt, np.abs(thr_cyc))
        if verbose:
            print('thr_pt', thr_pt)
            print('thr_cyc', thr_cyc)
            print('thr', thr)

        ncols = len(syss)
        fig, ax = plt.subplots(1, ncols, figsize=(ncols*5, 5), tight_layout=False, constrained_layout=True)
        fig2, ax2 = plt.subplots(1, ncols, figsize=(ncols*5, 5), tight_layout=False, constrained_layout=True)
        
        if ncols == 1:
            ax = [ax]
            ax2 = [ax2]

        # here, exp is a single repeat
        # output = self.exp_outputs[exp]
        # instead, compute mean over repeats of exp

        # fit_value = {sys: np.mean([self.exp_outputs[exp_rep][sys] for exp_rep in self.exp_repeats[exp]], axis=0) for sys in syss}
        fit_value = {}
        for sys in syss:
            fit_value[sys] = np.mean([np.sign(self.exp_outputs[exp_rep][sys]) for exp_rep in self.exp_repeats[exp] if sys in self.exp_outputs[exp_rep].keys()], axis=0)
            
        # fit_value = {sys: np.mean([np.sign(self.exp_outputs[exp_rep][sys]) , axis=0) for sys in syss if sys in self.exp_outputs[exp_rep].keys()}
        # fit_value = {sys: np.mean([np.sign(self.exp_outputs[exp_rep][sys]) for exp_rep in self.exp_repeats[exp]], axis=0) for sys in syss}


        for isys, sys in enumerate(syss):
            DE = SystemFamily.get_generator(self.ode_syss[sys])
            param_idx1 = DE.plot_param_idx[0]
            param_idx2 = DE.plot_param_idx[1] if len(DE.plot_param_idx) > 1 else param_idx1

            bifurp = self.bifurps[sys]
            sysp = self.sysps[sys]

            y_cyc = fit_value[sys][:,cyc_attr_idx]
            tit = f'{self.sys_descs[sys]}'

            if bifurp is not None:
                corr = np.corrcoef(bifurp, y_cyc)[0,1]
                print(sys, corr)
                tit = f'{self.sys_descs[sys]} ({corr:.2f})'

            param_descs = DE.param_descs
            
            # tit = f'{titles[sys]}\n distance from {bifurp_desc}' if isys == 0 else
            xlabel=param_descs[param_idx1]
            ylabel=param_descs[param_idx2]
            colorlabel = ''
            # if sysp.shape[1]>1:
                
            plot_diverge_scale(sysp[:,param_idx1], sysp[:,param_idx2], y_cyc, xlabel=xlabel, 
                                ylabel=ylabel, colorlabel=colorlabel, title=tit, ax=ax[isys], 
                                add_colorbar=False, labelsize=20, s=s, vmin=-1, vmax=1)
            
            x = sysp[:,param_idx1]
            y = sysp[:,param_idx2]
            
            z = y_cyc

            xy_grid, z_grid = get_prediction(x,y,z, npts=40, **kwargs_pred)
            
            tit = self.sys_descs[sys]
            plot_diverge_scale(xy_grid[:,0], xy_grid[:,1], z_grid, xlabel=xlabel, ylabel=ylabel, 
                                colorlabel=colorlabel, title=tit, ax=ax2[isys], add_colorbar=True, 
                                labelsize=20, s=50)
            
            # highlight pts of minimum value
            idx = np.where(np.abs(z_grid) < thr)
            ax2[isys].scatter(xy_grid[idx,0], xy_grid[idx,1], c='k', s=50)

            # adding true bifurcation boundaries
            lw = 3
            
            bifur_dfs = DE.get_bifurcation_curve()
            if bifur_dfs is not None:
                for bifur_df in bifur_dfs:
                    ax[isys].plot(bifur_df.values[:,param_idx1], bifur_df.values[:,param_idx2], color='k', lw=lw)


            # if sys.startswith('simple_oscillator'): 
            #     ax[isys].axvline(0, color='k', lw=lw)
            # if sys == 'suphopf':
            #     ax[isys].axvline(0, color='k', lw=lw)
            # if sys == 'lienard_poly':
            #     ax[isys].axvline(0, color='k', lw=lw)
            # if sys == 'lienard_sigmoid':
            #     ax[isys].axvline(0, color='k', lw=lw)
            # # bzreaction
            # if sys == 'bzreaction':
            #     a = np.arange(2,19,1); f = lambda a: 3*a/5 - 25/a
            #     ax[isys].plot(a, f(a), color='k', lw=lw)
            #     ax[isys].set_ylim(2, 6)
            #     # ax2[isys].plot(a, f(a), color='k', lw=lw)
            #     # ax2[isys].set_ylim(2, 6)

            #     # selkov
            # if sys == 'selkov':
            #     a = np.arange(.01,.11,0.01); 
            #     f_plus = lambda a: np.sqrt(1/2 * (1-2*a + np.sqrt(1-8*a)))
            #     f_minus = lambda a: np.sqrt(1/2 * (1-2*a - np.sqrt(1-8*a)))        
            #     ax[isys].plot(a, f_plus(a), color='k', lw=lw)
            #     ax[isys].plot(a, f_minus(a), color='k', lw=lw)
                
        plt.locator_params(axis='both', nbins=4)

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

