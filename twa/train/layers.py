import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.utils import weight_norm, spectral_norm
from typing import Tuple, Callable

class CNN(nn.Module):
    def __init__(self, in_shape, num_conv_layers=1,
            kernel_sizes=[], kernel_features=[],
            pooling_sizes = [],
            strides = [1],
            batch_norm=False, dropout=False, dropout_rate=.5,
            activation_type='relu'):

        super(CNN, self).__init__()

        if activation_type == 'relu':
            activation = torch.nn.ReLU()
        elif activation_type == 'leakyrelu':
            activation = torch.nn.LeakyReLU()
        elif activation_type == 'softplus':
            activation = torch.nn.Softplus()
        elif activation_type == 'tanh':
            activation = torch.nn.Tanh()
        elif activation_type == None:
            activation = None
        else:
            raise ValueError('Activation type not recognized!')

        conv_layers = []

        for l in range(num_conv_layers):
            in_channels = in_shape[0] if l==0 else kernel_features[l-1]
            conv_layers.append(torch.nn.Conv2d(in_channels, kernel_features[l], kernel_sizes[l], stride=strides[l]))
            if batch_norm:
                conv_layers.append(torch.nn.BatchNorm2d(kernel_features[l]))
            if activation != None:
               conv_layers.append(activation)
            if len(pooling_sizes) >= l +1:
                conv_layers.append(torch.nn.MaxPool2d(pooling_sizes[l]))

        self.conv_layers = torch.nn.Sequential(*conv_layers)

        with torch.no_grad():
            x = torch.rand(1, *in_shape)
            y = self.conv_layers(x)
            self.out_shape = y.shape[1:]
            self.out_size = torch.prod(torch.tensor(y.shape))

    def forward(self, x):
        return self.conv_layers(x)


class dCNN(nn.Module):
    def __init__(self, in_shape, num_conv_layers=1,                                                                                                                                                         
            kernel_sizes=[], kernel_features=[],                                                                                                                                                            
            pooling_sizes = [],                                                                                                                                                                             
            strides = [1],                                                                                                                                                                                  
            batch_norm=False, dropout=False, dropout_rate=.5,                                                                                                                                               
            activation_type='relu', last_pad=False):                                                                                                                                                        
                                                                                                                                                                                                            
        super(dCNN, self).__init__()                                                                                                                                                                        
                                                                                                                                                                                                            
        self.last_pad = last_pad                                                                                                                                                                            
                                                                                                                                                                                                            
        if activation_type == 'relu':                                                                                                                                                                       
            activation = torch.nn.ReLU()
        elif activation_type == 'leakyrelu':
            activation = torch.nn.LeakyReLU()
        elif activation_type == 'softplus':
            activation = torch.nn.Softplus()
        elif activation_type == 'tanh':
            activation = torch.nn.Tanh()
        elif activation_type == None:
            activation = None
        else:
            raise ValueError('Activation type not recognized!')

        deconv_layers = []
        for l in range(num_conv_layers):
            in_channels = in_shape[0] if l==0 else kernel_features[l-1]
            output_padding = 1 if l == (num_conv_layers - 1) and last_pad else 0
            deconv_layers.append(torch.nn.ConvTranspose2d(in_channels, kernel_features[l], kernel_sizes[l], stride=strides[l],output_padding=output_padding))
            if batch_norm:
                deconv_layers.append(torch.nn.BatchNorm2d(kernel_features[l]))
            if activation != None:
               deconv_layers.append(activation)
            if len(pooling_sizes) >= l +1:
                deconv_layers.append(torch.nn.MaxPool2d(pooling_sizes[l]))

        self.num_conv_layers = num_conv_layers
        self.deconv_layers = torch.nn.Sequential(*deconv_layers)

        with torch.no_grad():
            x = torch.rand(1, *in_shape)
            y = self.deconv_layers(x)
            self.out_shape = y.shape[1:]
            self.out_size = torch.prod(torch.tensor(y.shape))

    def forward(self, x):
        return self.deconv_layers(x)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_hid_layers=1, hid_dims=[64], batch_norm=False, normalize=None, dropout=False, dropout_rate=.5, activation_type='relu'):
        super(MLP, self).__init__()

        if activation_type == 'relu':
            activation = torch.nn.ReLU()
        elif activation_type == 'leakyrelu':
            activation = torch.nn.LeakyReLU()
        elif activation_type == 'softplus':
            activation = torch.nn.Softplus()
        elif activation_type == 'tanh':
            activation = torch.nn.Tanh()
        elif activation_type == None:
            activation = None
        else:
            raise ValueError('Activation type not recognized!')

        layers = []
        normalization = lambda x: x
        if normalize == 'weight':
            normalization = weight_norm
        elif normalize == 'spectral':
            normalization = spectral_norm #SpectralNorm
        
        for l in range(num_hid_layers + 1):
            if l == 0:
                if num_hid_layers == 0:
                    layers.append(torch.nn.Linear(in_dim, out_dim))
                else:
                    # if normalization is not None:
                    layers.append(normalization(torch.nn.Linear(in_dim, hid_dims[l])))#, dim=None))
                    # else:
                    #     layers.append(torch.nn.Linear(in_dim, hid_dims[l]))
                    if batch_norm:
                        layers.append(torch.nn.BatchNorm1d(hid_dims[l]))
                    if activation !=  None:
                        layers.append(activation)
                    if dropout:
                        layers.append(torch.nn.Dropout(p=dropout_rate))
            if l > 0:
                if l == num_hid_layers:
                    layers.append(torch.nn.Linear(hid_dims[l-1], out_dim))
                else:
                    # if add_weight_norm:
                    layers.append(normalization(torch.nn.Linear(hid_dims[l-1], hid_dims[l])))#, dim=None))
                    # else:
                    #     layers.append(torch.nn.Linear(hid_dims[l-1], hid_dims[l]))
                    if batch_norm:
                        layers.append(torch.nn.BatchNorm1d(hid_dims[l]))
                    if activation !=  None:
                        layers.append(activation)
                    if dropout:
                        layers.append(torch.nn.Dropout(p=dropout_rate))
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self,x):
        return self.layers(x)




class SelfAttention(nn.Module):
    """ Self attention Layer
    Editted slightly from: https://github.com/heykeetae/Self-Attention-GAN/
    """
    def __init__(self,in_dim, activation='softmax', width=None, height=None): 
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        if activation == 'softmax':
            self.activation = nn.Softmax(dim=-1) 
        elif activation == 'normalize':
            self.activation = nn.LayerNorm((width*height), elementwise_affine=False) 
        else:
            raise ValueError('Activation type not recognized!')

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1, width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        # attention = self.softmax(energy) # B X (N) X (N) 
        attention = self.activation(energy)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out, attention
    


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """ Spectral Norm
    Copied from: https://github.com/heykeetae/Self-Attention-GAN/
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)