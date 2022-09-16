"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.spectral_norm as spectral_norm
from model.blocks.sync_batchnorm import SynchronizedBatchNorm2d
from model.acon import *
from torch.nn import utils

def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        #print(param_free_norm_type)
        self.ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False, track_running_stats=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        self.nhidden = 128

        self.pw = self.ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, self.nhidden, kernel_size=self.ks, padding=self.pw),
            MetaAconC(self.nhidden),
        )
        self.mlp_gamma = nn.Conv2d(norm_nc, norm_nc, kernel_size=self.ks, padding=self.pw)
        self.mlp_beta = nn.Conv2d(norm_nc, norm_nc, kernel_size=self.ks, padding=self.pw)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        gamma, beta = self.get_spade_gamma_beta(normalized, segmap)
        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

    def get_spade_gamma_beta(self, normed, segmap):
        segmap = F.interpolate(segmap, size=normed.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return gamma, beta


class SPADE_Shoutcut(SPADE):
    def __init__(self, config_text,in_size, norm_nc, label_nc, opt, spade_mode="concat", use_en_feature=False, dil=False):
        super(SPADE_Shoutcut, self).__init__(config_text, norm_nc, label_nc)
        self.spade_mode = spade_mode
        self.dil = dil
        self.label_nc = label_nc
        if spade_mode == 'concat':
            # concat + conv
            self.con_conv = nn.Conv2d(norm_nc * 2, norm_nc, kernel_size=self.ks, padding=self.pw)
        if use_en_feature:
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(label_nc * 2, self.nhidden, kernel_size=self.ks, padding=self.pw),
                MetaAconC(self.nhidden)
            )
        self.metaAcon = MetaAconC(in_size)
        self.mlp_gamma = nn.Conv2d(norm_nc, norm_nc, kernel_size=self.ks, padding=self.pw)
        self.mlp_beta = nn.Conv2d(norm_nc, norm_nc, kernel_size=self.ks, padding=self.pw)
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)

    def forward(self, x, segmap, opt, en_feature=None, de_conv_actv=None, gamma_mode='none'):
        normalized = self.param_free_norm(x)
        gamma = self.mlp_gamma(de_conv_actv)
        beta = self.mlp_beta(de_conv_actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        if self.spade_mode == 'concat':
            concating = torch.cat([normalized, out], dim=1)
            out = self.con_conv(concating)
        elif self.spade_mode == 'res':
            out = out + normalized
        elif self.spade_mode == 'res2':
            out = out + x

        if gamma_mode == 'final':
            gamma_beta = torch.cat([gamma, beta], dim=1)  # use to calc l1
            print(gama_beta.shape)
            return out, gamma_beta
        elif gamma_mode == 'feature':
            norm_gamma = normalized.running_mean
            norm_beta = normalized.running_var
            gamma_beta = torch.cat([norm_gamma, norm_beta], dim=1)  # use to calc l1
            return out, gamma_beta
        else:
            return out


