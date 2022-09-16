import torch.nn as nn
import torch
from model.blocks.spade_normalization_acon_de2_dcn import SPADE_Shoutcut
from model.acon import *
import re
import torch.nn.functional as F
from model.DCNv2.dcn_v2 import DCN
from torch.nn import utils
import numpy as np
import json
import cv2
import torchvision
import os
import matplotlib.pyplot as plt

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, norm_fun=nn.BatchNorm2d):
        super(UNetDown, self).__init__()
        layers = [MetaAconC(in_size)]
        if normalize:
            if norm_fun == utils.spectral_norm:
                layers.append(utils.spectral_norm(nn.Conv2d(in_size, out_size, 4, 2, 1)))
            else:
                layers.append(nn.Conv2d(in_size, out_size, 4, 2, 1))
                layers.append(norm_fun(out_size, track_running_stats=False))
        else:
            layers.append(nn.Conv2d(in_size, out_size, 4, 2, 1))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class SPADEUp(nn.Module):
    def __init__(self, opt, in_size, out_size, dropout=0.0, first=False):
        super(SPADEUp, self).__init__()
        self.opt = opt
        parsing_nc = opt.parsing_nc
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.nhidden = out_size
        parsed = re.search('spade(\D+)(\d)x\d', spade_config_str)
        self.ks = int(parsed.group(2))
        self.pw = self.ks // 2
        self.layers = nn.ConvTranspose2d(in_size, out_size, 4, 2, 1)
        self.dcn = DCN(in_size, out_size, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
        self.fc1 = nn.Conv2d(parsing_nc * 2 + 3, self.nhidden, kernel_size=self.ks, padding=self.pw)
        self.fc2 = nn.Conv2d(parsing_nc + 3, self.nhidden, kernel_size=self.ks, padding=self.pw)
        self.fc3 = nn.Conv2d(self.nhidden,self.nhidden, kernel_size=self.ks, padding=self.pw)
        self.p1 = nn.Parameter(torch.randn(1, out_size, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, out_size, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.out_size = out_size
        if not first:
            self.norm = SPADE_Shoutcut(spade_config_str, in_size, out_size, parsing_nc, opt, opt.spade_mode,
                                       opt.use_en_feature)
        else:
            self.norm = SPADE_Shoutcut(spade_config_str, in_size, out_size, parsing_nc, opt, opt.spade_mode)
        self.en_conv = nn.ConvTranspose2d(in_size // 2, parsing_nc, 4, 2, 1)
        self.dp = None
        if dropout:
            self.dp = nn.Dropout(dropout)
        self.metaAcon = MetaAconC(out_size)
        self.metaAcon_en = MetaAconC(parsing_nc)
        self.param_free_norm = nn.InstanceNorm2d(out_size, affine=False, track_running_stats=False)

    def onehot_style_FS2K(self,img,names):
        b,c,h,w = img.shape
        if(isinstance(names, list) == False):
            result = []
            result.append(names)
            names = result
        path_train = "/data/yifan/FS2K/anno_train.json"
        path_test = "/data/yifan/FS2K/anno_test.json"
        temp = np.zeros([b,3, h, w])
        file_train = open(path_train, "rb")
        file_test = open(path_test, "rb")
        fileJson_train = json.load(file_train)
        fileJson_test = json.load(file_test)
        for i, name in enumerate(names):
            if (self.opt.isTrain or self.opt.style == 0):#FS2K数据集按照官方定的风格
                for li in fileJson_train:
                    name_raw = li["image_name"]
                    items = name_raw.split('/')
                    dir1 = items[0]
                    dir2 = items[1]
                    name_new = dir1[-1] + "_" + dir2 + ".jpg"
                    if(name_new == name):
                        style = li["style"]
                        if(style == 1):# style2
                            temp[i,0,:,:]=1
                        elif(style == 2):# style3
                            temp[i,1,:,:]=1
                        else:# style1
                            temp[i,2,:,:]=1
                        break
                for li in fileJson_test:
                    name_raw = li["image_name"]
                    items = name_raw.split('/')
                    dir1 = items[0]
                    dir2 = items[1]
                    name_new = dir1[-1] + "_" + dir2 + ".jpg"
                    if(name_new == name):
                        style = li["style"]
                        if (style == 1):# style2
                            temp[i,0,:,:] = 1
                        elif (style == 2):# style3
                            temp[i,1,:,:] = 1
                        else:# style1
                            temp[i,2,:,:] = 1
                        break
            else:#FS2K数据集测试时自己决定的风格
                if(self.opt.style == 1):#风格1
                    temp[ i,2,:,:] = 1
                elif(self.opt.style == 2):#风格2
                    temp[ i,0,:,:] = 1
                elif (self.opt.style == 3):#风格3
                    # print("3")
                    temp[ i,1,:,:] = 1
        temp = torch.from_numpy(temp).cuda()
        temp = torch.cat([img,temp], dim = 1)
        temp = temp.type(torch.cuda.FloatTensor)

        return temp

    def forward(self, de_in, parsing, en_in=None, gamma_mode='none', use_spade=False,  dcn = False, name = ""):
        x = de_in
        en_affine = None
        if en_in is not None:
            x = torch.cat([de_in, en_in], dim=1)
            if self.opt.use_en_feature:
                en_affine = self.en_conv(en_in)
                en_affine = self.metaAcon_en(en_affine)
        y1 = F.interpolate(x, scale_factor=2, mode='bilinear')
        if (dcn == True):
            y = self.dcn(y1)
        else:
            y = self.layers(x)

        if (self.opt.FS2K == True):
            parsing = self.onehot_style_FS2K(parsing, name)
            parsing = F.interpolate(parsing, size=[y.shape[2], y.shape[2]], mode='nearest')

        if en_in is not None:
            parsing = torch.cat([parsing, en_affine], dim=1)
            de_conv = self.fc1(parsing)
        else:
            de_conv = self.fc2(parsing)

        de_conv_actv = self.metaAcon(de_conv)
        beta = self.sigmoid(self.fc3(de_conv_actv))  # 0826
        de_act = (self.p1 * y - self.p2 * y) * self.sigmoid( beta * (self.p1 * y - self.p2 * y)) + self.p2 * y

        if (use_spade == False):
            z = self.param_free_norm(de_act)
        else:
            z = self.norm(de_act, parsing, self.opt, en_affine, de_conv_actv, gamma_mode=gamma_mode)

        if self.dp is not None:
            z = self.dp(z)
        return z
