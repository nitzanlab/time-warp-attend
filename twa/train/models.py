# prerequisites
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from twa.train.layers import MLP, CNN, dCNN, SelfAttention, SpectralNorm
from twa.train.utils import permute_data, rev_permute_data
from twa.data import sindy_library, FlowSystemODE, topo_num_to_str_dict, topo_point_vs_cycle
import matplotlib.pyplot as plt


latent_dim_default = 10
dropout_rate_default = 0.9
kernel_size_default = 3
conv_dim_default = 64
nconv_layers_default = 4


def load_model(model_type, pretrained_path=None, device='cpu', **kwargs):                                                                                                                                   
    """load_model: loads a neural network which maps from flows to either flows or parameters                                                                                                               
                                                                                                                                                                                                            
       Positional arguments:                                                                                                                                                                                
                                                                                                                                                                                                            
           model_type (str): name of model (see below)                                                                                                                                                      
           data_dim (int): number of dimensionin the phase space (i.e. number of array channels in data)
           num_lattice (int): number of points in the side of the cubic mesh on which velocities are measured in the data
           latent_dim (int): dimension of latent space

        Keyword arguments:
            num_DE_params (int): number of parameters in underlying system. Relevant for "Par" models (see below)
             
            """
     
    if model_type == 'AttentionwFC' or model_type is None:
        model = AttentionwFC_classify(**kwargs)
    elif model_type == 'FC':
        model = FC_classify(**kwargs)
    elif model_type == 'AE':
        model = AE(**kwargs)
    elif model_type == 'AEFC':
        model = AEFC_classify(**kwargs)
    elif model_type == 'CNNwFC_exp_emb' or model_type == 'p2v':
        model = CNNwFC_exp_emb(**kwargs)
    else:
        raise ValueError("Haven't added other models yet!")

    if pretrained_path is not None:
        if not os.path.isfile(pretrained_path):
            # try relative to running directory
            pretrained_path = os.path.join(os.getcwd(), pretrained_path)
            if not os.path.isfile(pretrained_path):
                raise ValueError('Invalid pretrained path: {}'.format(pretrained_path))
        
        model.load_state_dict(torch.load(pretrained_path, map_location=torch.device(device)))
    return model.to(device)


class AttentionwFC_classify(nn.Module):
    """
    Taken from Discriminator implementation at:
    https://github.com/heykeetae/Self-Attention-GAN
    """
    def __init__(self, in_shape, out_shape, latent_dim=latent_dim_default, with_attention=True, dropout_rate=dropout_rate_default, 
                 kernel_size=kernel_size_default, conv_dim=conv_dim_default, nconv_layers=nconv_layers_default, **kwargs):
        
        super(AttentionwFC_classify, self).__init__()
        self.in_shape = in_shape
        self.imsize = self.in_shape[-1]
        self.with_attention = with_attention
        in_channels = self.in_shape[0]
        self.nconv_layers = nconv_layers
        layer1 = []
        layer2 = []
        
        emb = []
        last = []

        normalization = SpectralNorm
        # normalization = weight_norm
        stride = 2
        padding = 1
        layer1.append(normalization(nn.Conv2d(in_channels, conv_dim, kernel_size, stride, padding)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(normalization(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size, stride, padding)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2


        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)

        if self.nconv_layers > 2:
            layer3 = []
            layer3.append(normalization(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size, stride, padding)))
            layer3.append(nn.LeakyReLU(0.1))
            curr_dim = curr_dim * 2
            self.l3 = nn.Sequential(*layer3)

        if self.nconv_layers > 3:
            layer4 = []
            layer4.append(normalization(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size, stride, padding)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim * 2

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, self.imsize, self.imsize)
            dummy = self.l1(dummy)
            l1_out_size = dummy.shape[1]
            dummy = self.l2(dummy)
            l2_out_size = dummy.shape[1]
            if self.nconv_layers > 2:
                dummy = self.l3(dummy)
                l3_out_size = dummy.shape[1]
            if self.nconv_layers > 3:
                dummy = self.l4(dummy)
                l4_out_size = dummy.shape[1]
            dummy = dummy.view(-1)
            cnn_out_size = dummy.shape[0]
        
        self.cnn_out_size = cnn_out_size
        self.emb = MLP(self.cnn_out_size, latent_dim, dropout=True, dropout_rate=dropout_rate) # , add_weight_norm=True
        self.last = nn.Linear(latent_dim, out_shape,)
        if self.with_attention:
            if self.nconv_layers == 2:
                self.attn1 = SelfAttention(l1_out_size, 'softmax')
                self.attn2 = SelfAttention(l2_out_size, 'softmax')
            if self.nconv_layers == 3:
                self.attn1 = SelfAttention(l2_out_size, 'softmax')
                self.attn2 = SelfAttention(l3_out_size, 'softmax')
            if self.nconv_layers == 4:
                self.attn1 = SelfAttention(l3_out_size, 'softmax')
                self.attn2 = SelfAttention(l4_out_size, 'softmax')
            

    def encode_cnn(self, x):
        p1 = p2 = None
        if self.nconv_layers == 2 and not self.with_attention:
            out = self.l1(x)
            out = self.l2(out)
        elif self.nconv_layers == 2 and self.with_attention:
            out = self.l1(x)
            out,p1 = self.attn1(out)
            out = self.l2(out)
            out,p2 = self.attn2(out)
        elif self.nconv_layers == 3 and not self.with_attention:
            out = self.l1(x)
            out = self.l2(out)
            out = self.l3(out)
        elif self.nconv_layers == 3 and self.with_attention:
            out = self.l1(x)
            out = self.l2(out)
            out,p1 = self.attn1(out)
            out = self.l3(out)
            out,p2 = self.attn2(out)
        elif self.nconv_layers == 4 and not self.with_attention:
            out = self.l1(x)
            out = self.l2(out)
            out = self.l3(out)
            out = self.l4(out)
        elif self.nconv_layers == 4 and self.with_attention:
            out = self.l1(x)
            out = self.l2(out)
            out = self.l3(out)
            out,p1 = self.attn1(out)
            out = self.l4(out)
            out,p2 = self.attn2(out)
        return out, [p1,p2]
    
    def encode(self, x):
        out, _ = self.encode_cnn(x)
        out = self.emb(out.view(-1, self.cnn_out_size)).squeeze()
        # out = self.emb(out.reshape(-1, self.cnn_out_size)).squeeze()
        return out
    
    def forward(self, x):
        out = self.encode(x)
        out=self.last(out)
        return out.squeeze()

    def plot_attention(self, data, atten_layer='p1', n_samples=3, device='cuda', tit=None):
        """
        Plot attention maps for a given model and data.
        """
        if not self.with_attention:
            return
        
        to_angle = self.in_shape[0] == 1
        imsize = self.imsize
        coords = np.array(np.meshgrid(np.arange(imsize), np.arange(imsize))).T[..., [1,0]]
        
        ncols = 3
        nrows = n_samples // ncols 
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*5))
        ax = ax.flatten()
        idx = np.arange(n_samples)
        X = data.data[idx].to(device)
        label_batch = data.label[idx].to(device)
        out, attns = self.encode_cnn(X)

        attn = attns[0] if atten_layer == 'p1' else attns[1]

        attn_size = int(np.sqrt(attn.shape[-1]))
        fold = imsize // attn_size
        
        label_tit = list(topo_point_vs_cycle().columns)
        
        if not to_angle:
            X = rev_permute_data(X)
        
        for i in range(n_samples):
            x = X[i]
            label = label_batch[i].cpu().detach().numpy()
            atten = attn[i, :].cpu().detach().numpy().sum(axis=0) # summing over attention values

            tit = ' & '.join([label_tit[tp] for tp in np.where(label)[0]])
            
            if to_angle:
                FlowSystemODE.plot_angle_image_(angle=x[0].cpu().detach().numpy(), title=tit, ax=ax[i])

            else:
                vectors = x.detach().cpu().numpy()
                FlowSystemODE.plot_trajectory_2d_(coords=coords, vectors=vectors, title=tit, ax=ax[i])
                

            atten = np.repeat(np.repeat(atten.reshape(attn_size, attn_size), fold, axis=0), fold, axis=1)
            # might need interpolation if not a multiple
            ax[i].imshow(atten.reshape(imsize, imsize), cmap='gray', alpha=0.5)
            ax[i].invert_yaxis()
            
            ax[i].axis('off')



class FC_classify(nn.Module):
    """
    Basic FC net
    """
    def __init__(self, in_shape, out_shape, **kwargs):
        
        super(FC_classify, self).__init__()
        if isinstance(in_shape, list):
            if len(in_shape) > 1:
                raise ValueError('FC expecting flattened input')
            else:
                in_shape = in_shape[0]
        self.in_shape = in_shape
        self.last = nn.Linear(self.in_shape, out_shape,)

    def forward(self, x):
        out = self.last(x)
        return out.squeeze()


class AE(nn.Module):
    def __init__(self, in_shape, latent_dim=latent_dim_default,
            num_conv_layers=3, kernel_sizes=3*[kernel_size_default], kernel_features=3*[128],
            pooling_sizes=[],
            strides = 3*[2],
            batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu',
            finetune=False, **kwargs):

        super(AE, self).__init__()
        self.finetune = finetune
        self.dim         = in_shape[0]
        self.grid_dim    = len(in_shape) - 1
    
        self.latent_dim = latent_dim
        self.enc_cnn = CNN(in_shape, num_conv_layers=num_conv_layers,
                        kernel_sizes=kernel_sizes, kernel_features=kernel_features,
                        pooling_sizes = pooling_sizes,
                        strides = strides,
                        batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                        activation_type=activation_type)
        

        self.out_shape = self.enc_cnn.out_shape
        self.out_size = self.enc_cnn.out_size
        

        self.emb_mlp   = MLP(self.out_size, latent_dim,
                         num_hid_layers=0, hid_dims=[],
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)

        self.deemb_mlp = MLP(latent_dim, self.out_size,
                         num_hid_layers=0, hid_dims=[],
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)

        deconv_kernel_features = kernel_features[::-1][:-1] + [self.dim]
        
        last_pad = (kernel_sizes[-1]-1) // 2

        self.dec_dcnn = dCNN(self.out_shape, num_conv_layers=num_conv_layers,
                           kernel_sizes=kernel_sizes[::-1], kernel_features=deconv_kernel_features,
                           pooling_sizes = pooling_sizes[::-1],
                           strides = strides,
                           batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                           activation_type=activation_type,
                           last_pad=last_pad)

    def forward(self,x):
        return self.decode(self.encode(x))

    def encode(self, x):
        x = self.enc_cnn(x[:, :self.dim,...])
        x = self.emb_mlp(x.reshape(-1, self.out_size))
        return x

    def decode(self, z):
        return self.dec_dcnn(self.deemb_mlp(z).reshape(-1, *self.out_shape))


class AEFC_classify(nn.Module):
    """
    Model composed of some encoder and a classifier
    """
    def __init__(self, model_ae, model_cl, **kwargs):
        
        super(AEFC_classify, self).__init__()
        self.model_ae = model_ae
        self.model_ae.requires_grad_ = False
        self.model_cl = model_cl
        self.encode = self.model_ae.encode
        self.classify = self.model_cl.forward
        

    def forward(self, x):
        out = self.encode(x)
        out = self.classify(out)
        return out.squeeze()
    
class CNNwFC_exp_emb(nn.Module):
    """
    Phase2vec architecture
    """
    def __init__(self, in_shape, poly_order=3, latent_dim=64,
            num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
            pooling_sizes=[],
            strides = [1],
            num_fc_hid_layers=0, fc_hid_dims=[],
            min_dims=[-1.,-1.],
            max_dims=[1.,1.],
            batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu', last_pad=None, **kwargs):

        super(CNNwFC_exp_emb, self).__init__()

        self.dim         = in_shape[0]
        self.num_lattice = in_shape[1]
        self.latent_dim = latent_dim

        encoder_model = CNN # if self.dim == 2 else CNN_nd
        self.enc = encoder_model(in_shape, num_conv_layers=num_conv_layers,
                                 kernel_sizes=kernel_sizes, kernel_features=kernel_features,
                                 pooling_sizes = pooling_sizes,
                                 strides = strides,
                                 batch_norm=batch_norm,
                                 activation_type=activation_type)
        with torch.no_grad():
            x = torch.rand(1, *in_shape)
            y = self.enc(x)
            self.out_shape = y.shape[1:]
            self.out_size = torch.prod(torch.tensor(y.shape))

        self.emb = MLP(self.out_size, latent_dim,
                         num_hid_layers=0, hid_dims=[],
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)

        self.poly_order = poly_order

        spatial_coords = [np.linspace(mn, mx, self.num_lattice) for (mn, mx) in zip(min_dims, max_dims)]
        mesh = np.meshgrid(*spatial_coords)
        self.L = torch.tensor(np.concatenate([ms[..., None] for ms in mesh], axis=-1))
        
        library, library_terms = sindy_library(self.L.reshape(self.num_lattice**self.dim,self.dim), poly_order)
        self.library = library.float()

        self.dec = MLP(latent_dim, self.dim * len(library_terms),
                       num_hid_layers=num_fc_hid_layers, hid_dims=fc_hid_dims,
                       batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                       activation_type=activation_type)

    def encode(self, x):
        x = self.enc(x)
        x = x.reshape(-1, self.out_size)
        return self.emb(x)

    def forward(self,x):
        return self.dec(self.encode(x))