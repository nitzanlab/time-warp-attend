import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch import nn
from twa.data import FlowSystemODE, topo_num_to_str_dict, topo_point_vs_cycle, topo_dont_know
from twa.train.utils import permute_data, rev_permute_data


num_all_classes = len(topo_num_to_str_dict)


class VecTopoDataset(Dataset):
    
    def __init__(self, data_dir, tt='train', num_classes=2, datatype='angle', datasize=None, noise=0.0, mask_prob=0.0, filter_outbound=False, dim=2, min_dims=[-1,-1], max_dims=[1,1]):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        
        """

        X = np.load(os.path.join(data_dir, f'X_{tt}.npy'))
        # self.X = torch.tensor(X).permute(0, 3, 1, 2).type(torch.FloatTensor)
        # TODO:temp
        # for i in range(dim):
        #     X[...,i] = X[...,i] / X[...,i].max(axis=(1,2), keepdims=True)
        
        # rescale to maximum norm of 1 per vector
        # if datatype == 'vector':
        #     X = X / np.linalg.norm(X, axis=-1, keepdims=True)
        #     X[np.where(np.isnan(X))] = 0.
        
        self.X = permute_data(torch.tensor(X)) if X.shape[1] != dim else torch.tensor(X)
        self.angle = torch.tensor(np.arctan2(X[...,1], X[...,0])) 
        self.angle = self.angle.unsqueeze(1)
        self.num_classes = num_classes
        self.num_lattice = self.X.shape[-1]
        self.coords = np.array(np.meshgrid(np.arange(self.num_lattice), np.arange(self.num_lattice))).T[..., [1,0]]
        
        self.label = None
        self.sysp = None
        self.params = None
        self.fixed_pts = None

        fname = os.path.join(data_dir, f'sysp_{tt}.npy')
        if os.path.isfile(fname):
            self.sysp = np.load(fname)

        fname = os.path.join(data_dir, f'topo_{tt}.npy')
        if os.path.isfile(fname):
            topo = np.load(fname)
            self.label = topo_point_vs_cycle(topo)
            self.label_tit = list(self.label.columns)
            self.label = self.label.values
            self.label = torch.tensor(self.label).type(torch.IntTensor)

        self.dim = dim
        self.max_dims = max_dims
        self.min_dims = min_dims

        self.datasize = len(self.X)

        
        fname = os.path.join(data_dir, f'p_{tt}.npy')
        if os.path.isfile(fname): 
            self.params = torch.tensor(np.load(fname))

        fname = os.path.join(data_dir, f'fixed_pts_{tt}.npy')
        if os.path.isfile(fname):
            self.fixed_pts = np.load(fname)
            if filter_outbound:
                idx = FlowSystemODE.get_pts_isin_(self.fixed_pts, min_dims=self.min_dims, max_dims=self.max_dims)
                self.angle = self.angle[idx]
                self.X = self.X[idx]
                self.label = self.label[idx] if self.label is not None else None
                self.sysp = self.sysp[idx] if self.sysp is not None else None
                self.params = self.params[idx] if self.params is not None else None
                self.fixed_pts = self.fixed_pts[idx]  if self.fixed_pts is not None else None
                self.datasize = len(self.X)

        print(self.datasize)
        
        
        if datasize is not None:
            new_datasize = min(datasize, self.datasize)

            if new_datasize < self.datasize:
                idx = np.random.choice(self.datasize, new_datasize, replace=False)
                self.angle = self.angle[idx]
                self.X = self.X[idx]
                self.label = self.label[idx] if self.label is not None else None
                self.sysp = self.sysp[idx] if self.sysp is not None else None
                self.params = self.params[idx] if self.params is not None else None
                self.fixed_pts = self.fixed_pts[idx]
            elif self.datasize < new_datasize:
                print(f'WARNING: datasize {self.datasize} < {new_datasize}') 


        # selecting data
        self.datatype = datatype
        self.data = None
        if self.datatype == 'angle':
            self.data = self.angle
        elif self.datatype == 'vector':
            self.data = self.X
        elif self.datatype == 'param':
            self.data = self.params

        if self.data is None:
            raise ValueError(f'Datatype {self.datatype} is not available.')
        self.data = self.data.type(torch.FloatTensor)

        # adding noises
        if noise > 0:
            self.data += torch.randn_like(self.data) * noise
            if datatype == 'angle':
                self.data = self.data.clamp_(-np.pi, np.pi)
        if mask_prob > 0:
            # self.data[torch.rand_like(self.data) < mask_prob] = 0
            idx_mask = torch.rand(self.datasize, self.num_lattice, self.num_lattice) < mask_prob
            if datatype == 'angle':
                idx_mask = torch.unsqueeze(idx_mask, dim=1)
            elif datatype == 'vector':
                idx_mask = torch.stack((idx_mask, idx_mask), dim=1)
            self.data[idx_mask] = 0


            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        label = topo_dont_know if self.label is None else self.label[idx]
        return [data, label]
        
    
    def plot_data(self, num_samples=9, sample_random=True):
        nrows = 3
        ncols = int(num_samples/nrows)
        fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
        ax = ax.flatten()
        num_samples = len(ax)
        idx = np.random.choice(self.__len__(), num_samples, replace=False) if sample_random else np.arange(num_samples)
        
        datas = self.data[idx]
        labels = self.label[idx]
        if self.datatype == 'vector':
            datas = rev_permute_data(datas)

        for i, (x,label) in enumerate(zip(datas, labels)):
            if i >= num_samples:
                break
            
            tit = ' & '.join([self.label_tit[tp] for tp in np.where(label)[0]])
            if self.datatype == 'angle':
                FlowSystemODE.plot_angle_image_(angle=x[0], title=tit, ax=ax[i])
            elif self.datatype == 'vector':
                vectors = x.detach().numpy()
                FlowSystemODE.plot_trajectory_2d_(coords=self.coords, vectors=vectors, title=tit, ax=ax[i])
            elif self.datatype == 'param':
                DE = Polynomial(x)
                DE.plot_trajectory(title=tit, ax=ax[i])
                
                # cb = fig.colorbar(stream.lines)
                # cb.cla() 
            ax[i].axis('off')
            
        plt.show()


