# prerequisites
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from .layers import MLP, CNN, SelfAttention, SpectralNorm
from .utils import permute_data, rev_permute_data
from twa.data import sindy_library, FlowSystemODE, topo_num_to_str_dict, topo_point_vs_cycle #topo_attr_spiral, topo_period_attr
import matplotlib.pyplot as plt


# def load_model(model_type, pretrained_path=None, device='cpu', **kwargs):
#     """load_model: loads a neural network which maps from flows to either flows or parameters

#        Positional arguments:

#            model_type (str): name of model (see below)
#            data_dim (int): number of dimensionin the phase space (i.e. number of array channels in data)
#            num_lattice (int): number of points in the side of the cubic mesh on which velocities are measured in the data
#            latent_dim (int): dimension of latent space

#         Keyword arguments:
#             num_DE_params (int): number of parameters in underlying system. Relevant for "Par" models (see below)
            
#             """
    
#     if model_type == 'CNNwFC_exp_emb':
#         model = CNNwFC_exp_emb(device=device, **kwargs)
#     # elif model_type == 'Conv2dAE':
#     #     model = Conv2dAE(**kwargs)
#     # elif model_type == 'AELIC':
#     #     model = AELIC(**kwargs)
#     # elif model_type == 'VAE_exp_emb':
#     #     model = VAE_exp_emb(device=device, **kwargs)
#     # # elif model_type == 'FC_VAE':
#     # #     model = FC_VAE(device=device, **kwargs)
#     else:
#         raise ValueError("Haven't added other models yet!")

#     if pretrained_path is not None:
#         model.load_state_dict(torch.load(pretrained_path, map_location=torch.device(device)))
#     return model.to(device)


# class CNNwFC_exp_emb(nn.Module):
#     def __init__(self, in_shape, poly_order=3, latent_dim=64,
#             num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
#             pooling_sizes=[],
#             strides = [1],
#             num_fc_hid_layers=0, fc_hid_dims=[],
#             min_dims=[-1.,-1.],
#             max_dims=[1.,1.],
#             added_channels=0, w_channels=0.3,
#             device='cpu',
#             batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu', **kwargs):

#         super(CNNwFC_exp_emb, self).__init__()
#         self.added_channels = added_channels
#         self.grid_dim         = len(in_shape) - 1 #
#         self.dim = in_shape[0] 
#         assert(self.dim == (self.grid_dim + self.added_channels))
#         assert(self.grid_dim == len(min_dims))
#         assert(self.grid_dim == len(max_dims))
        
#         self.num_lattice = in_shape[1]

#         self.latent_dim = latent_dim
#         self.isAE = False
#         self.w_channels = w_channels if self.added_channels > 0 else 0

#         # CNN encoder: vectors + other channels -> convolutions
#         self.enc_cnn = CNN(in_shape, num_conv_layers=num_conv_layers,
#                         kernel_sizes=kernel_sizes, kernel_features=kernel_features,
#                         pooling_sizes = pooling_sizes,
#                         strides = strides,
#                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                         activation_type=activation_type)

#         # one run for output shape
#         self.out_shape = self.enc_cnn.out_shape
#         self.out_size = self.enc_cnn.out_size
        
#         # fully connected embedding: convolutions -> latent space
#         self.emb_mlp = MLP(self.out_size, latent_dim,
#                          num_hid_layers=0, hid_dims=[],
#                          batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                          activation_type=activation_type)

#         # decoder
#         self.poly_order = poly_order

#         spatial_coords = [np.linspace(mn, mx, self.num_lattice) for (mn, mx) in zip(min_dims, max_dims)]
#         mesh = np.meshgrid(*spatial_coords)
#         self.L = torch.tensor(np.concatenate([ms[..., None] for ms in mesh], axis=-1))
        
#         library, library_terms = sindy_library(self.L.reshape(self.num_lattice**self.grid_dim, self.grid_dim), poly_order)
#         self.library = library.float().to(device)

#         # parameters (midway of parameter decoder): latent space -> parameter estimation
#         self.pars_mlp = MLP(latent_dim, self.grid_dim * len(library_terms),
#                        num_hid_layers=0, hid_dims=[],
#                        batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                        activation_type=activation_type)

#         # parameter decoder: latent space -> vector reconstruction
             
#         # added channels decoder: latent space -> added channels reconstruction
#         if self.added_channels > 0:
#             self.added_channels_dec_mlp = MLP(latent_dim, self.out_size,
#                           num_hid_layers=0, hid_dims=[],   
#                             batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                             activation_type=activation_type)

#             deconv_kernel_features = kernel_features[::-1][:-1] + [self.added_channels]
        
#             last_pad = (kernel_sizes[-1]-1) // 2
#             self.added_channels_dec_dcnn = dCNN(self.out_shape, num_conv_layers=num_conv_layers,
#                             kernel_sizes=kernel_sizes[::-1], kernel_features=deconv_kernel_features,
#                             pooling_sizes = pooling_sizes[::-1],
#                             strides = strides,
#                             batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                             activation_type=activation_type, last_pad=last_pad)

#     def forward(self,x):
#         return self.decode(self.encode(x))

#     def encode(self, x):
#         """
#         Encode input x to latent space (z)
#         """
#         x = self.enc_cnn(x)
#         x = x.reshape(-1, self.out_size)
#         return self.emb_mlp(x)

#     def decode_pars(self, z):
#         """
#         Decode latent space to parameters
#         """
#         pars = self.pars_mlp(z).reshape(-1, self.library.shape[-1], self.grid_dim)
#         return pars
            
#     def decode_added_channels(self, z):
#         """
#         Decode added channels to reconstruction
#         """
#         x = self.added_channels_dec_mlp(z).reshape(-1, *self.out_shape)
#         x = self.added_channels_dec_dcnn(x)
        
#         return x

#     def decode(self, z):
#         """
#         Decode latent space to reconstruction
#         """
#         pars = self.decode_pars(z)
#         recon = permute_data(torch.einsum('sl,bld->bsd', self.library, pars).reshape(-1, self.num_lattice, self.num_lattice, self.grid_dim)) #.permute(0,3,1,2) #TODO: via permute data
        
#         if self.added_channels > 0:
#             return torch.concat((recon, 
#                                  self.decode_added_channels(z)), axis=1) # should I split the embedding, no! 
#         else:
#             return recon











# class Conv2dAE(nn.Module):
#     def __init__(self, in_shape, latent_dim=64,
#             num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
#             pooling_sizes=[],
#             strides = [1],
#             num_fc_hid_layers=0, fc_hid_dims=[],
#             batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu',
#             added_channels=0, w_channels=0.5,
#             finetune=False, min_dims=None, max_dims=None, poly_order=None, **kwargs):

#         super(Conv2dAE, self).__init__()
#         self.added_channels = added_channels
#         self.finetune = finetune
#         self.dim         = in_shape[0]
#         self.grid_dim    = len(in_shape) - 1
#         self.w_channels = w_channels if self.added_channels > 0 else 0

#         assert(self.dim == (self.grid_dim + self.added_channels))
#         self.latent_dim = latent_dim
#         self.enc_subchannels_cnn = CNN(in_shape, num_conv_layers=num_conv_layers,
#                         kernel_sizes=kernel_sizes, kernel_features=kernel_features,
#                         pooling_sizes = pooling_sizes,
#                         strides = strides,
#                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                         activation_type=activation_type)
        
#         self.isAE = True

#         self.out_shape = self.enc_subchannels_cnn.out_shape
#         self.out_size = self.enc_subchannels_cnn.out_size
        

#         self.emb_mlp   = MLP(self.out_size, latent_dim,
#                          num_hid_layers=0, hid_dims=[],
#                          batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                          activation_type=activation_type)

#         self.deemb_mlp = MLP(latent_dim, self.out_size,
#                          num_hid_layers=0, hid_dims=[],
#                          batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                          activation_type=activation_type)

#         deconv_kernel_features = kernel_features[::-1][:-1] + [self.dim]
        
#         last_pad = (kernel_sizes[-1]-1) // 2

#         self.dec_dcnn = dCNN(self.out_shape, num_conv_layers=num_conv_layers,
#                            kernel_sizes=kernel_sizes[::-1], kernel_features=deconv_kernel_features,
#                            pooling_sizes = pooling_sizes[::-1],
#                            strides = strides,
#                            batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                            activation_type=activation_type,
#                            last_pad=last_pad)

#     # def forward(self,x):
#     #     x = self.conv1(x)
#     #     if self.finetune:
#     #         with torch.no_grad():
#     #             x = self.enc(x)
#     #     else:
#     #         x = self.enc(x)
#     #     return self.dec(x)

#     def forward(self,x):
#         return self.decode(self.encode(x))

#     def encode(self, x):
#         """
#         Encode input to latent space
#         """
#         x = self.enc_subchannels_cnn(x[:, :self.dim,...])
#         x = self.emb_mlp(x.reshape(-1, self.out_size))
#         return x

#     def decode(self, z):
#         """
#         Decode latent space to reconstruction
#         """
#         return self.dec_dcnn(self.deemb_mlp(z).reshape(-1, *self.out_shape))



# class VAE_exp_emb(nn.Module):
#     def __init__(self, in_shape, poly_order=3, latent_dim=64,
#             num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
#             pooling_sizes=[],
#             strides = [1],
#             num_fc_hid_layers=0, fc_hid_dims=[],
#             min_dims=[-1.,-1.],
#             max_dims=[1.,1.],
#             added_channels=0, w_channels=0.2,
#             device='cpu',
#             batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu', **kwargs):

#         super(VAE_exp_emb, self).__init__()
#         self.added_channels = added_channels
#         self.grid_dim         = len(in_shape) - 1 #
#         self.dim = in_shape[0] 
#         assert(self.dim == (self.grid_dim + self.added_channels))
#         assert(self.grid_dim == len(min_dims))
#         assert(self.grid_dim == len(max_dims))
        
#         self.num_lattice = in_shape[1]

#         self.latent_dim = latent_dim
#         self.isAE = False
#         self.w_channels = w_channels if self.added_channels > 0 else 0

#         # CNN encoder: vectors + other channels -> convolutions
#         self.enc_cnn = CNN(in_shape, num_conv_layers=num_conv_layers,
#                         kernel_sizes=kernel_sizes, kernel_features=kernel_features,
#                         pooling_sizes = pooling_sizes,
#                         strides = strides,
#                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                         activation_type=activation_type)

#         # one run for output shape
#         self.out_shape = self.enc_cnn.out_shape
#         self.out_size = self.enc_cnn.out_size
        
#         # fully connected embedding: convolutions -> mu in latent space
#         self.emb_mlp_mu = MLP(self.out_size, latent_dim,
#                          num_hid_layers=0, hid_dims=[],
#                          batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                          activation_type=activation_type)

#         # fully connected embedding: convolutions -> mu in latent space
#         self.emb_mlp_log_var = MLP(self.out_size, latent_dim,
#                          num_hid_layers=0, hid_dims=[],
#                          batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                          activation_type=activation_type)

#         self.N = torch.distributions.Normal(0, 1)
#         if device == 'cuda':
#             self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
#             self.N.scale = self.N.scale.cuda()
#         self.kl = 0


#         # decoder
#         self.poly_order = poly_order

#         spatial_coords = [np.linspace(mn, mx, self.num_lattice) for (mn, mx) in zip(min_dims, max_dims)]
#         mesh = np.meshgrid(*spatial_coords)
#         self.L = torch.tensor(np.concatenate([ms[..., None] for ms in mesh], axis=-1))
        
#         library, library_terms = sindy_library(self.L.reshape(self.num_lattice**self.grid_dim, self.grid_dim), poly_order)
#         self.library = library.float().to(device)

#         # sampling from latent space


#         # parameters (midway of parameter decoder): latent space -> parameter estimation
#         self.pars_mlp = MLP(latent_dim, self.grid_dim * len(library_terms),
#                        num_hid_layers=0, hid_dims=[],
#                        batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                        activation_type=activation_type)

#         # parameter decoder: latent space -> vector reconstruction
             
#         # added channels decoder: latent space -> added channels reconstruction
#         if self.added_channels > 0:
#             self.added_channels_dec_mlp = MLP(latent_dim, self.out_size,
#                           num_hid_layers=0, hid_dims=[],   
#                             batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                             activation_type=activation_type)

#             deconv_kernel_features = kernel_features[::-1][:-1] + [self.added_channels]
        
#             last_pad = (kernel_sizes[-1]-1) // 2
#             self.added_channels_dec_dcnn = dCNN(self.out_shape, num_conv_layers=num_conv_layers,
#                             kernel_sizes=kernel_sizes[::-1], kernel_features=deconv_kernel_features,
#                             pooling_sizes = pooling_sizes[::-1],
#                             strides = strides,
#                             batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                             activation_type=activation_type, last_pad=last_pad)

#     def forward(self,x):
#         # mu, log_var = self.encode(x)
#         # z = self.sample(mu, log_var)
#         z = self.encode(x)
#         return self.decode(z)

#     def encode(self, x):
#         """
#         Encode input x to latent space (z)
#         """
#         x = self.enc_cnn(x)
#         x = x.reshape(-1, self.out_size)
#         mu = self.emb_mlp_mu(x)
#         log_var = self.emb_mlp_log_var(x)

#         var = torch.exp(log_var)
        
#         z = mu + var*self.N.sample(mu.shape)
#         self.kl = (var**2 + mu**2 - torch.log(var) - 1/2).sum()
#         return z

#     # def sample(self, mu, log_var):
#     #     """
#     #     Sample from latent space
#     #     """
#     #     var = torch.exp(log_var)
        
#     #     z = mu + var*self.N.sample(mu.shape)
#     #     self.kl = (var**2 + mu**2 - torch.log(var) - 1/2).sum()
#     #     return z

#     def decode_pars(self, z):
#         """
#         Decode latent space to parameters
#         """
#         pars = self.pars_mlp(z).reshape(-1, self.library.shape[-1], self.grid_dim)
#         return pars
            
#     def decode_added_channels(self, z):
#         """
#         Decode added channels to reconstruction
#         """
#         x = self.added_channels_dec_mlp(z).reshape(-1, *self.out_shape)
#         x = self.added_channels_dec_dcnn(x)
        
#         return x

#     def decode(self, z):
#         """
#         Decode latent space to reconstruction
#         """
#         pars = self.decode_pars(z)
#         recon = permute_data(torch.einsum('sl,bld->bsd', self.library, pars).reshape(-1, self.num_lattice, self.num_lattice, self.grid_dim)) #.permute(0,3,1,2) #TODO: via permute data
        
#         if self.added_channels > 0:
#             return torch.concat((recon, self.decode_added_channels(z)), axis=1) # should I split the embedding, no! 
#         else:
#             return recon



# class AELIC(nn.Module):
#     """
#     (Temporary) autoencoder of lic channel only, adding a random variable to the latent space
#     """
#     def __init__(self, in_shape, latent_dim=4,
#             num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
#             pooling_sizes=[],
#             strides = [1],
#             num_fc_hid_layers=0, fc_hid_dims=[],
#             batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu',
#             added_channels=1, w_channels=1, 
#             finetune=False, min_dims=None, max_dims=None, poly_order=None, **kwargs):

#         super(AELIC, self).__init__()
#         self.added_channels = added_channels
#         self.finetune = finetune
#         self.dim         = in_shape[0]
#         self.grid_dim    = len(in_shape) - 1

#         assert(self.dim == (self.grid_dim + self.added_channels))
#         self.latent_dim = latent_dim
#         channels_shape = (self.added_channels, *in_shape[1:])
#         self.w_channels = w_channels
#         self.enc_subchannels_cnn = CNN(channels_shape, num_conv_layers=num_conv_layers,
#                         kernel_sizes=kernel_sizes, kernel_features=kernel_features,
#                         pooling_sizes = pooling_sizes,
#                         strides = strides,
#                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                         activation_type=activation_type)
        
#         # taking only channels in channels

#         self.isAE = True

#         self.out_shape = self.enc_subchannels_cnn.out_shape
#         self.out_size = self.enc_subchannels_cnn.out_size
        
#         self.emb_mlp   = MLP(self.out_size, latent_dim,
#                          num_hid_layers=0, hid_dims=[],
#                          batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                          activation_type=activation_type)

#         self.deemb_mlp = MLP(latent_dim, self.out_size,
#                          num_hid_layers=0, hid_dims=[],
#                          batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                          activation_type=activation_type)

#         deconv_kernel_features = kernel_features[::-1][:-1] + [self.dim]
        
#         last_pad = (kernel_sizes[-1]-1) // 2

#         self.dec_dcnn = dCNN(self.out_shape, num_conv_layers=num_conv_layers,
#                            kernel_sizes=kernel_sizes[::-1], kernel_features=deconv_kernel_features,
#                            pooling_sizes = pooling_sizes[::-1],
#                            strides = strides,
#                            batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                            activation_type=activation_type,
#                            last_pad=last_pad)
            

#     # def forward(self,x):
#     #     x = self.conv1(x)
#     #     if self.finetune:
#     #         with torch.no_grad():
#     #             x = self.enc(x)
#     #     else:
#     #         x = self.enc(x)
#     #     return self.dec(x)

#     def forward(self,x):
#         return self.decode(self.encode(x))

#     def encode(self, x):
#         """
#         Encode input to latent space
#         """
#         x = self.enc_subchannels_cnn(x[:, -self.added_channels:,...])
#         x = self.emb_mlp(x.reshape(-1, self.out_size))
#         return x

#     def decode(self, z):
#         """
#         Decode latent space to reconstruction
#         """
#         return self.dec_dcnn(self.deemb_mlp(z).reshape(-1, *self.out_shape))



# # class VariationalEncoder(nn.Module):
# #     def __init__(self, in_shape, latent_dim, device='cpu'):
# #         super(VariationalEncoder, self).__init__()
# #         self.in_shape = in_shape
# #         self.latent_dim = latent_dim
# #         self.in_flatten = np.prod(in_shape)
# #         self.linear1 = nn.Linear(self.in_flatten, 512)
# #         self.linear2 = nn.Linear(512, latent_dim)
# #         self.linear3 = nn.Linear(512, latent_dim)

# #         self.N = torch.distributions.Normal(0, 1)
# #         if device == 'cuda':
# #             self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
# #             self.N.scale = self.N.scale.cuda()
# #         self.kl = 0

    
# #     def forward(self, x, noise=True):
# #         x = torch.flatten(x, start_dim=1)
# #         x = F.relu(self.linear1(x))
# #         mu =  self.linear2(x)
# #         if noise:
# #             sigma = torch.exp(self.linear3(x))
# #             z = mu + sigma*self.N.sample(mu.shape)
# #             self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
# #         else:
# #             z = mu
# #         return z
    
# # class Decoder(nn.Module):
# #     def __init__(self, in_shape, latent_dim):
# #         super(Decoder, self).__init__()
# #         self.in_shape = in_shape
# #         self.latent_dim = latent_dim
# #         self.in_flatten = np.prod(in_shape)
        
# #         self.linear1 = nn.Linear(latent_dim, 512)
# #         self.linear2 = nn.Linear(512, self.in_flatten)

# #     def forward(self, z):
# #         z = F.relu(self.linear1(z))
# #         z = torch.sigmoid(self.linear2(z))
# #         return z.reshape((-1, *self.in_shape))
    
    
# # class FC_VAE(nn.Module):
# #     def __init__(self, in_shape, latent_dim, device='cpu', **kwargs):
# #         super(FC_VAE, self).__init__()
        
# #         self.grid_dim = len(in_shape) - 1
# #         self.added_channels = 0
# #         self.encoder = VariationalEncoder(in_shape=in_shape, latent_dim=latent_dim, device=device)
# #         self.decoder = Decoder(in_shape=in_shape, latent_dim=latent_dim)
# #         self.kl = 0

# #     def encode(self, x, noise=True):
# #         z = self.encoder(x, noise=noise)
# #         self.kl = self.encoder.kl
# #         return z
        
# #     def decode(self, z):
# #         return self.decoder(z)

# #     def forward(self, x):
# #         z = self.encode(x)
# #         return self.decode(z)


# class CNNwFC_classify(nn.Module):
#     def __init__(self, in_shape, out_shape, latent_dim=64,
#             num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
#             pooling_sizes=[],
#             strides = [1],
#             num_fc_hid_layers=0, fc_hid_dims=[],
#             min_dims=[-1.,-1.],
#             max_dims=[1.,1.],
#             device='cpu',
#             batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu', **kwargs):

#         super(CNNwFC_classify, self).__init__()
        
#         # CNN encoder: vectors + other channels -> convolutions
#         self.enc_cnn = CNN(in_shape, num_conv_layers=num_conv_layers,
#                         kernel_sizes=kernel_sizes, kernel_features=kernel_features,
#                         pooling_sizes = pooling_sizes,
#                         strides = strides,
#                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                         activation_type=activation_type)

#         # one run for output shape
#         self.out_shape = self.enc_cnn.out_shape
#         self.out_size = self.enc_cnn.out_size
        
#         # fully connected embedding: convolutions -> latent space
#         self.emb_mlp = MLP(self.out_size, latent_dim,
#                          num_hid_layers=0, hid_dims=[],
#                          batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#                          activation_type=activation_type)

    
#         # parameters (midway of parameter decoder): latent space -> classification
#         # self.pred_mlp = MLP(latent_dim, out_shape,
#         #                num_hid_layers=0, hid_dims=[],
#         #                batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
#         #                activation_type=activation_type)
#         self.pred_linear = nn.Linear(2*latent_dim, out_shape)

        
#     def forward(self,x):
#         return self.predict(self.encode(x))

#     def encode(self, x):
#         """
#         Encode input x to latent space (z)
#         """
#         x = self.enc_cnn(x)
#         x = x.reshape(-1, self.out_size)
#         return self.emb_mlp(x)

    
#     def predict(self, z):
#         """
#         Predict classes from latent space
#         """
#         z_z2 = torch.concat([z, z**2], dim=1)
#         # return self.pred_mlp(z)
#         return self.pred_linear(z_z2)
    



class AttentionwFC_classify(nn.Module):
    """
    Taken from Discriminator implementation at:
    https://github.com/heykeetae/Self-Attention-GAN
    """
    def __init__(self, in_shape, out_shape, latent_dim=10, with_attention=True, dropout_rate=0.9, kernel_size=3, conv_dim=64, nconv_layers=4):
        # num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
        # pooling_sizes=[],
        # strides = [1],
        # num_fc_hid_layers=0, fc_hid_dims=[],
        # min_dims=[-1.,-1.],
        # max_dims=[1.,1.],
        # device='cpu',
        # batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu', **kwargs):

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

        
        

        # if self.imsize == 64:
        #     layer4 = []
        #     layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size, stride, padding)))
        #     layer4.append(nn.LeakyReLU(0.1))
        #     self.l4 = nn.Sequential(*layer4)
        #     curr_dim = curr_dim*2
        
        
        # last.append(nn.Conv2d(curr_dim, out_shape, 4))
        # self.last = nn.Sequential(*last)

        # emb.append(nn.Conv2d(curr_dim, latent_dim, kernel_size))
        # self.emb = nn.Sequential(*emb)

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
            # self.attn1 = SelfAttention(l1_out_size, 'relu') 
            # self.attn2 = SelfAttention(l2_out_size, 'relu')
            if self.nconv_layers == 2:
                self.attn1 = SelfAttention(l1_out_size, 'softmax') #TODO: changed #'normalize', width=8, height=8
                self.attn2 = SelfAttention(l2_out_size, 'softmax') #'normalize', width=4, height=4
            if self.nconv_layers == 3:
                self.attn1 = SelfAttention(l2_out_size, 'softmax') #TODO: changed #'normalize', width=8, height=8
                self.attn2 = SelfAttention(l3_out_size, 'softmax') #'normalize', width=4, height=4
            if self.nconv_layers == 4:
                self.attn1 = SelfAttention(l3_out_size, 'softmax') #TODO: changed #'normalize', width=8, height=8
                self.attn2 = SelfAttention(l4_out_size, 'softmax') #'normalize', width=4, height=4
            # self.attn1 = SelfAttention(l3_out_size, 'softmax') #TODO: changed #'normalize', width=8, height=8
            # self.attn2 = SelfAttention(l4_out_size, 'softmax') #'normalize', width=4, height=4
            # self.attn1 = SelfAttention(l3_out_size, 'normalize', width=8, height=8) ##TODO: changed
            # self.attn2 = SelfAttention(l4_out_size, 'normalize', width=4, height=4)
    

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
        # out = self.l1(x)
        # if self.with_attention:
        #     out,p1 = self.attn1(out)
        # out = self.l2(out)
        # if self.with_attention:
        #     out,p1 = self.attn1(out)
        # out = self.l3(out)
        # if self.with_attention:
        #     out,p2 = self.attn2(out)
        # out=self.l4(out)
        # if self.with_attention:
        #     out,p2 = self.attn2(out)
        return out, [p1,p2]
    
    def encode(self, x):
        out, _ = self.encode_cnn(x)
        out=self.emb(out.view(-1, self.cnn_out_size)).squeeze()
        # out = self.emb(out.reshape(-1, self.cnn_out_size)).squeeze()
        return out
    
    def forward(self, x):
        out = self.encode(x)
        out=self.last(out)
        return out.squeeze()

    def plot_attention(self, data, atten_layer='p1', n_samples=3, device='cuda', tit=None):
        """Plot attention maps for a given model and data."""
        if not self.with_attention:
            return
        
        # model.eval()
        to_angle = self.in_shape[0] == 1
        imsize = self.imsize
        coords = np.array(np.meshgrid(np.arange(imsize), np.arange(imsize))).T[..., [1,0]]
        # ncols = 2 if to_angle else 3
        ncols = 3
        nrows = n_samples // ncols 
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*5))
        ax = ax.flatten()
        idx = np.random.randint(0, len(data), n_samples)
        X = data.data[idx].to(device)
        label_batch = data.label[idx].to(device)
        if not to_angle:
            X = rev_permute_data(X)
        # for X, label_batch in data:
        out, attns = self.encode_cnn(X)

        attn = attns[0] if atten_layer == 'p1' else attns[1]

        attn_size = int(np.sqrt(attn.shape[-1]))
        fold = imsize // attn_size

        # atten_pt = atten_pt_w * attn_size + atten_pt_h
        # num_all_classes = len(topo_num_to_str_dict)
        # topo_idx = [topo_attr_spiral, topo_period_attr]
        label_tit = list(topo_point_vs_cycle().columns)
        
        for i in range(n_samples):
            x = X[i]
            label = label_batch[i].cpu().detach().numpy()
            atten = attn[i, :].cpu().detach().numpy().sum(axis=0)

            # p1,p2 are attention maps of B x N x N dimensions where N = W x H
            # decide on a random point of interest p , 
            # reshape p1[i,atten_pt,:] to W x H and plot it

            # stretch attention map to the size of the image
            # tit = ''.join([(topo_num_to_str_dict[tp] if (label[itp] == 1) else '') for itp, tp in enumerate(topo_idx) ]) if tit is None else tit
            tit = ' & '.join([label_tit[tp] for tp in np.where(label)[0]])
            
            if to_angle:
                FlowSystemODE.plot_angle_image_(angle=x[0].cpu().detach().numpy(), title=tit, ax=ax[i])

            else:
                # vectors = x.permute(1,2,0).detach().cpu().numpy()
                vectors = x.detach().cpu().numpy()
                FlowSystemODE.plot_trajectory_2d_(coords=coords, vectors=vectors, title=tit, ax=ax[i])
                # ax[i].set_title(p2v.dt.topo_to_str(np.where(label.detach().numpy())[0]))        

            atten = np.repeat(np.repeat(atten.reshape(attn_size, attn_size), fold, axis=0), fold, axis=1)
            # might need interpolation if not a multiple
            ax[i].imshow(atten.reshape(imsize, imsize), cmap='gray', alpha=0.5)
            ax[i].invert_yaxis()
            
            ax[i].axis('off')




class AttentionwFC_classify_submission(nn.Module):
    """
    Taken from Discriminator implementation at:
    https://github.com/heykeetae/Self-Attention-GAN
    """
    def __init__(self, in_shape, out_shape, latent_dim, with_attention, dropout_rate=0.9, kernel_size=3, conv_dim=64):
        # num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
        # pooling_sizes=[],
        # strides = [1],
        # num_fc_hid_layers=0, fc_hid_dims=[],
        # min_dims=[-1.,-1.],
        # max_dims=[1.,1.],
        # device='cpu',
        # batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu', **kwargs):

        super(AttentionwFC_classify, self).__init__()
        self.in_shape = in_shape
        self.imsize = self.in_shape[-1]
        self.with_attention = with_attention
        in_channels = self.in_shape[0]
        
        layer1 = []
        layer2 = []
        layer3 = []
        emb = []
        last = []

        stride = 2
        padding = 1
        layer1.append(SpectralNorm(nn.Conv2d(in_channels, conv_dim, kernel_size, stride, padding)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size, stride, padding)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size, stride, padding)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2


        layer4 = []
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size, stride, padding)))
        layer4.append(nn.LeakyReLU(0.1))
        self.l4 = nn.Sequential(*layer4)
        curr_dim = curr_dim * 2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        

        # if self.imsize == 64:
        #     layer4 = []
        #     layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size, stride, padding)))
        #     layer4.append(nn.LeakyReLU(0.1))
        #     self.l4 = nn.Sequential(*layer4)
        #     curr_dim = curr_dim*2
        
        
        # last.append(nn.Conv2d(curr_dim, out_shape, 4))
        # self.last = nn.Sequential(*last)

        # emb.append(nn.Conv2d(curr_dim, latent_dim, kernel_size))
        # self.emb = nn.Sequential(*emb)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, self.imsize, self.imsize)
            dummy = self.l1(dummy)
            dummy = self.l2(dummy)
            dummy = self.l3(dummy)
            l3_out_size = dummy.shape[1]
            dummy = self.l4(dummy)
            l4_out_size = dummy.shape[1]
            dummy = dummy.view(-1)
            cnn_out_size = dummy.shape[0]
        
        self.cnn_out_size = cnn_out_size
        self.emb = MLP(self.cnn_out_size, latent_dim, dropout=True, dropout_rate=dropout_rate)
        self.last = nn.Linear(latent_dim, out_shape,)
        if self.with_attention:
            self.attn1 = SelfAttention(l3_out_size, 'relu') 
            self.attn2 = SelfAttention(l4_out_size, 'relu')
    
    
    def encode(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        if self.with_attention:
            out,_ = self.attn1(out)
        out=self.l4(out)
        if self.with_attention:
            out,_ = self.attn2(out)

        out=self.emb(out.view(-1, self.cnn_out_size)).squeeze()
        return out
    
    def forward(self, x):
        out = self.encode(x)
        out=self.last(out)
        return out.squeeze()

    def plot_attention(self, data, atten_layer='p1', n_samples=3, device='cuda'):
        """Plot attention maps for a given model and data."""
        if not self.with_attention:
            return
        
        # model.eval()
        to_angle = self.in_shape[0] == 1
        imsize = self.imsize
        coords = np.array(np.meshgrid(np.arange(imsize), np.arange(imsize))).T[..., [1,0]]
        # ncols = 2 if to_angle else 3
        ncols = 3
        nrows = n_samples // ncols 
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*5))
        ax = ax.flatten()
        idx = np.random.randint(0, len(data), n_samples)
        X = data.data[idx].to(device)
        label_batch = data.label[idx].to(device)
        # for X, label_batch in data:
        out = self.l1(X.to(device))
        # print(f'L1 output: {out.shape}')
        out = self.l2(out)
        # print(f'L2 output: {out.shape}')
        out = self.l3(out)
        out,p1 = self.attn1(out)
        # print(f'L3 output: {out.shape}')
        # out,p1 = self.attn1(out)
        # print(f'Atten output: {out.shape}')
        # print(f'Atten mask: {p1.shape}')
        # out = self.l4(out)
        # out,p2 = self.attn2(out)

        attens = p1 #if atten_layer == 'p1' else p2

        atten_size = int(np.sqrt(attens.shape[-1]))
        fold = imsize // atten_size

        # atten_pt = atten_pt_w * atten_size + atten_pt_h

        for i in range(n_samples):
            x = X[i]
            label = label_batch[i]
            atten = attens[i, :].cpu().detach().numpy().sum(axis=0)

            # p1,p2 are attention maps of B x N x N dimensions where N = W x H
            # decide on a random point of interest p , 
            # reshape p1[i,atten_pt,:] to W x H and plot it

            # stretch attention map to the size of the image
            # tit = ''.join([(topo_num_to_str_dict[tp] if (label[itp] == 1) else '') for itp, tp in enumerate(topo_idx) ])
            tit = ''
            if to_angle:
                FlowSystemODE.plot_angle_image_(x[0].cpu().detach().numpy(), title=tit, ax=ax[i])

            else:
                vectors = x.permute(1,2,0).detach().cpu().numpy()
                FlowSystemODE.plot_trajectory_2d_(coords=coords, vectors=vectors, title=tit, ax=ax[i])
                # ax[i].set_title(p2v.dt.topo_to_str(np.where(label.detach().numpy())[0]))        

            atten = np.repeat(np.repeat(atten.reshape(atten_size, atten_size), fold, axis=0), fold, axis=1)
            # might need interpolation if not a multiple
            ax[i].imshow(atten.reshape(imsize, imsize), cmap='gray', alpha=0.5)
            ax[i].invert_yaxis()
            
            ax[i].axis('off')


