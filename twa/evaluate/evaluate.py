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
from sklearn.metrics import silhouette_score
from twa.utils import plot_diverge_scale, get_prediction
import scanpy as sc
import scvelo as scv
torch.manual_seed(0)

sys_descs_default = {}
sys_descs_default['simple_oscillator_noaug'] = 'SO'
sys_descs_default['simple_oscillator_nsfcl'] = 'Augmented SO'
sys_descs_default['suphopf'] = 'Supercritical Hopf'
sys_descs_default['bzreaction'] = 'BZ Reaction'
sys_descs_default['selkov'] = 'Sel\'kov'
sys_descs_default['lienard_poly'] = 'Lienard Poly'
sys_descs_default['lienard_sigmoid'] = 'Lienard Sigmoid'
sys_descs_default['pancreas_clusters_random_bin'] = 'Pancreas'

ode_syss = {k: k for k in sys_descs_default.keys()}
ode_syss['simple_oscillator_noaug'] = 'simple_oscillator'
ode_syss['simple_oscillator_nsfcl'] = 'simple_oscillator'
ode_syss['pancreas_clusters_random_bin'] = None

sys_colors = {}
sys_colors['simple_oscillator_noaug'] = '#192BC2'
sys_colors['simple_oscillator_nsfcl'] = '#36C9C6'
sys_colors['suphopf'] = '#F5F749'
sys_colors['bzreaction'] = '#ED6A5A'
sys_colors['selkov'] = '#B30089'
sys_colors['lienard_poly'] = 'gray'
sys_colors['lienard_sigmoid'] = 'lightgray'
sys_colors['pancreas_clusters_random_bin'] = '#C879FF'

class Evaluator:
    def __init__(self, exp_descs, sys_descs=None, num_lattice=64, tt='test', outdir='../output/', verbose=False, repeats=np.inf):
        """
        
        """

        assert(isinstance(exp_descs, dict))
        self.exp_descs = exp_descs
        self.exps = list(exp_descs.keys())
        self.nexps = len(self.exps)

        self.outdir = outdir
        self.data_dir = os.path.join(outdir, 'data')
        self.result_dir = os.path.join(outdir, 'results')
        if (not os.path.isdir(self.outdir)) or (not os.path.isdir(self.data_dir)) or (not os.path.isdir(self.result_dir)):
            raise ValueError('Invalid output directory.')
        
        self.repeats = repeats
        self.num_lattice = num_lattice
        self.tt = tt
        
        # verify all ground truth syss exist
        self.sys_descs = sys_descs if sys_descs is not None else sys_descs_default
        for sys in self.sys_descs.keys():
            ddir = os.path.join(self.data_dir, sys)
            if not os.path.isdir(ddir):
                print(f'Ground truth for {sys} does not exist.')
                self.sys_descs.pop(sys)
        self.syss = list(self.sys_descs.keys())
        self.nsyss = len(self.syss)

        ### Read ground truth and results
        if verbose:
            print('Reading ground truth...')

        self.topos = {}
        self.sysps = {}
        self.dists = {}
        self.y_pts = {}
        self.bifurps = {}
        self.fixed_pts = {}

        for sys in self.syss:
            ddir = os.path.join(self.data_dir, sys)
            self.topos[sys] = np.load(os.path.join(ddir, f'topo_{tt}.npy'))
            self.sysps[sys] = np.load(os.path.join(ddir, f'sysp_{tt}.npy'))
            fname = os.path.join(ddir, f'dists_{tt}.npy')
            self.dists[sys] = np.load(fname).flatten() if os.path.isfile(fname) else None

            dist = self.dists[sys]
            dist = dist.reshape(-1) if dist is not None else None
            topo = self.topos[sys]
            
            label = topo_point_vs_cycle(topo)
            # assuming binary topology classification
            has_cyc = label['Periodic'].values
            self.y_pts[sys] = 1 - has_cyc
            self.bifurps[sys] = (-dist * (1 - has_cyc) + dist * has_cyc) if dist is not None else None

            # bad handling but read fixed pts and filter the data here if outside bounds
            fname = os.path.join(ddir, f'fixed_pts_{tt}.npy')
            if os.path.isfile(fname):
                fixed_pt = np.load(fname)
                self.fixed_pts[sys] = fixed_pt

        self.adata = None
        if 'pancreas_clusters_random_bin' in self.syss:
            self.adata = sc.read_h5ad(os.path.join(self.data_dir,'pancreas', 'adata.h5'))

        if verbose:
            print('Reading predictions...')

        # self.exp_yhats = {}
        self.exp_outputs = {}
        self.exp_pred_pts = {}
        self.exp_embeddings = {}
        self.exp_repeats = {}

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
                exp_result_dir = exp_rep_fname #os.path.join(self.result_dir, exp_rep)
                # if verbose:
                #     print(exp_rep_fname)
            
                exp_rep = os.path.basename(exp_rep_fname)
                exp_reps.append(exp_rep)
                # yhats = {}
                embeddings = {}
                outputs = {}
                pred_pts = {}
                for sys in self.syss:
                    # fname = os.path.join(exp_result_dir, sys, f'yhat_{tt}.npy')
                    # if os.path.isfile(fname):
                    #     yhats[sys] = np.load(fname)
                    fname = os.path.join(exp_result_dir, sys, f'embeddings_{tt}.npy')
                    if os.path.isfile(fname):
                        embeddings[sys] = np.load(fname)
                    fname = os.path.join(exp_result_dir, sys, f'outputs_{tt}.npy')
                    if os.path.isfile(fname):
                        outputs[sys] = np.load(fname)
                        pred_pts[sys] = outputs[sys][:, pt_attr_idx] > 0

                # self.exp_yhats[exp_rep] = yhats
                self.exp_embeddings[exp_rep] = embeddings
                self.exp_outputs[exp_rep] = outputs
                self.exp_pred_pts[exp_rep] = pred_pts
            self.exp_repeats[exp] = exp_reps

    
    def gather_acc_sens_spec(self):
        """
        Compute sensitivity, specificity, accuracy for point class. 
        """
        res = []
        # considering 0-true as point attractor, else, cycle
        for exp_rep in self.exp_pred_pts.keys():

            pred_pts = self.exp_pred_pts[exp_rep]
            
            for sys,pred_pt in pred_pts.items():
                y_pt = self.y_pts[sys]
                repeat = exp_rep.split('_')[-1]
                exp = '_'.join(exp_rep.split('_')[:-1])
                res.append( {'exp': exp, 'repeat':repeat, 'system': sys,
                            'Sens': np.round(np.mean(pred_pt[y_pt]), 2), 
                            'Spec': np.round(np.mean(1-pred_pt[~y_pt]), 2),
                            'Accu': np.round(np.mean(pred_pt==y_pt), 2)} )
                
        res = pd.DataFrame(res)
        return res
    
    def get_acc_sens_spec(self):
        """
        Get accuracy, sensitivity and specificity results.
        """
        res = self.gather_acc_sens_spec()
        df = pd.melt(res, id_vars=['exp', 'system'], value_vars=['Sens', 'Spec', 'Accu'])
        df = df.groupby(['exp', 'system', 'variable']).mean().pivot_table(index=['exp', 'variable'], columns='system', values='value')
        df.index.names = ['','']
        df.columns.name = ''
        df = df.loc[self.exps].rename(index=self.exp_descs)
        df = df[self.syss].rename(columns=self.sys_descs)
        return df
        

    def get_acc(self):
        """
        Get accuracy results.
        """
        res = self.gather_acc_sens_spec()
        df = res[['exp', 'system', 'Accu']].groupby(['exp', 'system']).mean().pivot_table(index='exp', columns='system', values='Accu')
        df.index.name = ''
        df.columns.name = ''
        df = df.loc[self.exps].rename(index=self.exp_descs)
        df = df[self.syss].rename(columns=self.sys_descs)
        
        return df

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
                
                ax[iexp].scatter(latent_pca[:,0], latent_pca[:,1], color=sys_colors[sys], s=50, alpha=0.5, label=self.sys_descs[sys])

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


    def plot_bifurcation(self, exp, selected_syss=None):
        """
        Plot confidence and inferred bifurcation diagrams.
        """
        selected_syss = self.syss if selected_syss is None else selected_syss
        selected_syss = [sys for sys in selected_syss if ode_syss[sys] is not None]
        ncols = len(selected_syss)
        fig, ax = plt.subplots(1, ncols, figsize=(ncols*5, 5), tight_layout=False, constrained_layout=True)
        fig2, ax2 = plt.subplots(1, ncols, figsize=(ncols*5, 5), tight_layout=False, constrained_layout=True)
        thr = 1
        output = self.exp_outputs[exp]
        

        for isys, sys in enumerate(selected_syss):
            DE = SystemFamily.get_generator(ode_syss[sys])
            param_idx1, param_idx2 = DE.plot_param_idx
            bifurp = self.bifurps[sys]
            sysp = self.sysps[sys]

            y_cyc = output[sys][:,cyc_attr_idx]

            corr = np.corrcoef(bifurp, y_cyc)[0,1]
            print(sys, corr)
        
            param_descs = DE.param_descs
            bifurp_desc = DE.bifurp_desc
            # tit = f'{titles[sys]}\n distance from {bifurp_desc}' if isys == 0 else
            xlabel=param_descs[param_idx1]
            ylabel=param_descs[param_idx2]
            colorlabel = ''
            if sysp.shape[1]>1:
                tit = f'{self.sys_descs[sys]} ({corr:.2f})'
                plot_diverge_scale(sysp[:,param_idx1], sysp[:,param_idx2], y_cyc, xlabel=xlabel, 
                                   ylabel=ylabel, colorlabel=colorlabel, title=tit, ax=ax[isys], 
                                   add_colorbar=False, labelsize=20, s=50)
                
                x = sysp[:,param_idx1]
                y = sysp[:,param_idx2]
                
                z = y_cyc

                xy_grid, z_grid = get_prediction(x,y,z, npts=40)
                
                tit = self.sys_descs[sys]
                plot_diverge_scale(xy_grid[:,0], xy_grid[:,1], z_grid, xlabel=xlabel, ylabel=ylabel, 
                                   colorlabel=colorlabel, title=tit, ax=ax2[isys], add_colorbar=True, 
                                   labelsize=20, s=50)
                
                # highlight pts of minimum value
                idx = np.where(np.abs(z_grid) < thr)
                ax2[isys].scatter(xy_grid[idx,0], xy_grid[idx,1], c='k', s=50)

            # adding true bifurcation boundaries
            lw = 3
            
            # bifur_dfs = DE.get_bifurcation_curve()
            # for bifur_df in bifur_dfs:
            #     ax[isys].plot(bifur_df.values[param_idx1], bifur_df.values[param_idx2], color='k', lw=lw)


            if sys.startswith('simple_oscillator'): 
                ax[isys].axvline(0, color='k', lw=lw)
            if sys == 'suphopf':
                ax[isys].axvline(0, color='k', lw=lw)
            if sys == 'lienard_poly':
                ax[isys].axvline(0, color='k', lw=lw)
            if sys == 'lienard_sigmoid':
                ax[isys].axvline(0, color='k', lw=lw)
            # bzreaction
            if sys == 'bzreaction':
                a = np.arange(2,19,1); f = lambda a: 3*a/5 - 25/a
                ax[isys].plot(a, f(a), color='k', lw=lw)
                ax[isys].set_ylim(2, 6)
                # ax2[isys].plot(a, f(a), color='k', lw=lw)
                # ax2[isys].set_ylim(2, 6)

                # selkov
            if sys == 'selkov':
                a = np.arange(.01,.11,0.01); 
                f_plus = lambda a: np.sqrt(1/2 * (1-2*a + np.sqrt(1-8*a)))
                f_minus = lambda a: np.sqrt(1/2 * (1-2*a - np.sqrt(1-8*a)))        
                ax[isys].plot(a, f_plus(a), color='k', lw=lw)
                ax[isys].plot(a, f_minus(a), color='k', lw=lw)
                
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
        exp_output = self.exp_outputs[exp][sys]
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

        plt.show()