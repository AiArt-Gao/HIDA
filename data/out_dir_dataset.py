# -*- coding: utf-8 -*-
from glob import glob
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from kornia import Resize
from kornia.augmentation import CenterCrop
from kornia.color import Normalize
# from kornia.enhance import Normalize
import kornia
from model.parsing.model import BiSeNet


class Outsketch_Folder_Set(data.Dataset):
    def __init__(self, opt: object, dirpath: object, use_par: object = True) -> object:
        '''
            need set opt.img_h and opt.img_w
        :param opt:
        :param dirpath:
        :param use_par:
        :param b_checkpoint_path:
        '''
        super(Outsketch_Folder_Set, self).__init__()
        self.datalist = sorted(glob("{}/*.jpg".format(dirpath))+glob("{}/*.png".format(dirpath)), key=os.path.getctime)
        self.lens = len(self.datalist)
        self.opt = opt
        self.use_par = use_par

    def __getitem__(self, index):
        line = self.datalist[index]
        inputs, name = self.loadImg(line)
        return {"inputs":inputs,"name": name}

    def __len__(self):
        return self.lens

    def loadImg(self, line):
        src = Image.open(line).convert("RGB")
        if self.use_par:
            par, name = self.get_depth(line)
            print(name)
        src = kornia.image_to_tensor(np.array(src, dtype=float)).float() / 255
        img = torch.cat([src, par])
        return img, name

    def get_depth(self,line):
        items = line.split('/')
        filename = items[-1]
        name = filename[:-4]
        depthPath = os.path.join(self.opt.depth_dir, name + '.jpg')
        depth = Image.open(depthPath).convert("L")
        depth = kornia.image_to_tensor(np.array(depth, dtype=float)).float()/255
        depth = torch.tensor(depth).float()
        gauss = kornia.filters.GaussianBlur2d(kernel_size=(5, 5), sigma=(0.2, 0.2))
        depth = depth.unsqueeze(dim=0)
        depth = gauss(depth)[0]

        return depth, filename

