import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch import nn
from twa.data.ode import FlowSystemODE
from twa.data.topology import topo_num_to_str_dict, topo_point_vs_cycle
from twa.train.utils import permute_data, rev_permute_data
# import torch.optim as optim
# from twa.utils import ensure_dir, write_yaml
# import random


num_all_classes = len(topo_num_to_str_dict)


class VecTopoDataset(Dataset):
    
    def __init__(self, data_dir, tt='train', num_classes=2, to_angle=True, datasize=None, noise=0.0, mask_prob=0.0, filter_outbound=False, dim=2, min_dims=[-1,-1], max_dims=[1,1]):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        X = np.load(os.path.join(data_dir, f'X_{tt}.npy'))
        topo = np.load(os.path.join(data_dir, f'topo_{tt}.npy'))

        # self.X = torch.tensor(X).permute(0, 3, 1, 2).type(torch.FloatTensor)
        self.X = permute_data(torch.tensor(X)).type(torch.FloatTensor)
        self.angle = torch.tensor(np.arctan2(X[...,1], X[...,0])).type(torch.FloatTensor) 
        self.angle = self.angle.unsqueeze(1)
        self.num_classes = num_classes
        self.num_lattice = self.X.shape[-1]
        self.coords = np.array(np.meshgrid(np.arange(self.num_lattice), np.arange(self.num_lattice))).T[..., [1,0]]
        
        self.label = topo_point_vs_cycle(topo)
        self.label_tit = list(self.label.columns)
        self.label = self.label.values
        self.label = torch.tensor(self.label).type(torch.IntTensor)

        self.dim = dim
        self.max_dims = max_dims
        self.min_dims = min_dims
        self.to_angle = to_angle

        if self.to_angle:
            self.data = self.angle
        else:
            self.data = self.X
        self.datasize = len(self.data)

        fname = os.path.join(data_dir, f'sysp_{tt}.npy')
        
        if os.path.isfile(fname):
            self.sysp = np.load(fname)
        
        fname = os.path.join(data_dir, f'p_{tt}.npy')
        if os.path.isfile(fname): 
            self.params = np.load(fname)

        self.fixed_pts = None
        fname = os.path.join(data_dir, f'fixed_pts_{tt}.npy')
        if os.path.isfile(fname):
            self.fixed_pts = np.load(fname)
            if filter_outbound:
                idx = FlowSystemODE.get_pts_isin_(self.fixed_pts, min_dims=self.min_dims, max_dims=self.max_dims)
                # idx = fixed_pt_isin[:,0,0] & fixed_pt_isin[:,0,1] 
                # frac_inbound = pt_in.sum() / pt_in.size * 100
                self.data = self.data[idx]
                self.angle = self.angle[idx]
                self.X = self.X[idx]
                self.label = self.label[idx]
                self.sysp = self.sysp[idx] if self.sysp is not None else None
                self.params = self.params[idx] if self.params is not None else None
                self.fixed_pts = self.fixed_pts[idx]  if self.fixed_pts is not None else None
                self.datasize = len(self.data)


        print(self.datasize)
        
        
        if datasize is not None:
            self.datasize = min(datasize, len(self.data)) 

            if self.datasize < len(self.data):
                idx = np.random.choice(len(self.data), self.datasize, replace=False)
                self.data = self.data[idx]
                self.angle = self.angle[idx]
                self.X = self.X[idx]
                self.label = self.label[idx]
                self.sysp = self.sysp[idx] if self.sysp is not None else None
                self.params = self.params[idx] if self.params is not None else None
                self.fixed_pts = self.fixed_pts[idx]
            elif self.datasize < datasize:
                print(f'WARNING: datasize {self.datasize} < {datasize}') 
            

        # adding noises
        if noise > 0:
            self.data += torch.randn_like(self.data) * noise
            if to_angle:
                self.data = self.data.clamp_(-np.pi, np.pi)
        if mask_prob > 0:
            self.data[torch.rand_like(self.data) < mask_prob] = 0

            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.label[idx]
        data = self.data[idx]
        
        return [data, label]
    
    # check data
    def plot_data(self, num_samples=9):
        nrows = 3
        ncols = int(num_samples/nrows)
        fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
        ax = ax.flatten()
        num_samples = len(ax)
        idx = np.random.choice(self.__len__(), num_samples, replace=False)
        
        datas = self.data[idx]
        labels = self.label[idx]
        if not self.to_angle:
            datas = rev_permute_data(datas)

        for i, (x,label) in enumerate(zip(datas, labels)):
            if i >= num_samples:
                break
            
            # tit = ''.join([(topo_num_to_str_dict[tp] if (label[itp] == 1) else '') for itp, tp in enumerate(topo_idx) ])
            tit = ' & '.join([self.label_tit[tp] for tp in np.where(label)[0]])
            if self.to_angle:
                FlowSystemODE.plot_angle_image_(angle=x[0], title=tit, ax=ax[i])
            else:
                # vectors = x.permute(1,2,0).detach().numpy()
                vectors = x.detach().numpy()
                FlowSystemODE.plot_trajectory_2d_(coords=self.coords, vectors=vectors, title=tit, ax=ax[i])
                
                # cb = fig.colorbar(stream.lines)
                # cb.cla() 
            ax[i].axis('off')
            
        # plot_angle_im(x[0], add_colorbar=True, ax=ax[i-1])
        plt.show()


