import os
import numpy as np
import torch
import random
import torch.utils.data as data
from os import path as osp
import scipy.io as sio
import kornia
from PIL import Image
from kornia.augmentation import RandomCrop
from glob import glob
import torchvision.transforms as transforms
import json

class Photosketch_Kornia_Set(data.Dataset):
    affine_type: object

    def __init__(self, opt, forTrain=True, use_par=True, ske_to_img=False, additional_bright=False):
        super(Photosketch_Kornia_Set, self).__init__()
        self.root = opt.root
        self.imgResol = opt.input_size
        self.ske_to_img = ske_to_img
        self.opt = opt
        if forTrain:
            dirpath = os.path.join(self.root, '/train/photos')
        else:
            dirpath = os.path.join(self.root, '/test/photos')

        self.datalist = sorted(glob("{}/*.jpg".format(dirpath)) + glob("{}/*.png".format(dirpath)),
                               key=os.path.getctime)
        self.lens = len(self.datalist)
        self.add_bright = additional_bright
        self.forTrain = forTrain
        self.use_par = use_par
        self.affine_type = opt.affine_type  # 对训练集做大型偏移的类型,normal
        self.img_nc = opt.image_nc
        self.depth_nc = opt.depth_nc
        self.par_nc = opt.parsing_nc
        self.tar_nc = opt.output_nc
        self.loaded_pool = {}  # 这里dataloader 不要开num_of_worker


    def __getitem__(self, index): #0106 +返回文件名
        line = self.datalist[index]
        tensors, name= self.loadImg(line)
        return tensors, name

    def __len__(self):
        return self.lens

    def apply_tranform(self, tensors):

        if self.forTrain:  # apply same Flip in source and target while training
            tensors = kornia.augmentation.RandomHorizontalFlip()(tensors)
        if len(tensors.size()) == 3:  # bsize = 1
            tensors = tensors.unsqueeze(0)

        # for x and y transform methods
        x_trans_arr = []
        y_trans_arr = []

        # share transform
        org_nc = self.img_nc
        if self.use_par:
            org_nc += self.par_nc

        y_trans_arr.append(kornia.geometry.Resize((self.opt.img_h, self.opt.img_w)))
        if self.forTrain:
            loadSize = int(np.ceil(self.opt.input_size * 1.117))
            if loadSize % 2 == 1:
                loadSize += 1
            if self.affine_type == "normal":
                y_trans_arr.append(kornia.augmentation.CenterCrop((loadSize, loadSize)))
            elif self.affine_type == "width":
                y_trans_arr.append(RandomCrop((loadSize, loadSize), pad_if_needed=True))
            elif self.affine_type == "scale":
                scale_factor = random.gauss(1, 0.15)
                if scale_factor > 1.3:
                    scale_factor = 1.3
                elif scale_factor < 0.7:
                    scale_factor = 0.7
                n_w = round(self.opt.img_w * scale_factor)
                n_h = round(self.opt.img_h * scale_factor)

                y_trans_arr.append(kornia.Resize((n_h, n_w)))
                y_trans_arr.append(kornia.augmentation.CenterCrop((loadSize, loadSize)))

            y_trans_arr.append(RandomCrop(size=(self.opt.input_size, self.opt.input_size), pad_if_needed=True))

            x_trans_arr.append(kornia.augmentation.CenterCrop((loadSize, loadSize)))
            x_trans_arr.append(RandomCrop(size=(self.opt.input_size, self.opt.input_size), pad_if_needed=True))
            if self.opt.FS2K:#FS2K
                y_trans_arr.append(kornia.filters.GaussianBlur2d(kernel_size=(5, 5), sigma=(0.2, 0.2)))  # 在模型上使用0均值1方差进行高斯模糊
                x_trans_arr.append(kornia.filters.GaussianBlur2d(kernel_size=(5, 5), sigma=(0.2, 0.2)))  # 在模型上使用0均值1方差进行高斯模糊

        else:  # test
            y_trans_arr.append(kornia.augmentation.CenterCrop((self.opt.input_size, self.opt.input_size)))
            x_trans_arr.append(kornia.augmentation.CenterCrop((self.opt.input_size, self.opt.input_size)))
            if self.opt.FS2K:#FS2K
                y_trans_arr.append(kornia.filters.GaussianBlur2d(kernel_size=(5, 5), sigma=(0.2, 0.2)))  # 在模型上使用0均值1方差进行高斯模糊
                x_trans_arr.append(kornia.filters.GaussianBlur2d(kernel_size=(5, 5), sigma=(0.2, 0.2)))  # 在模型上使用0均值1方差进行高斯模糊

        y_trans_method = torch.nn.Sequential(*y_trans_arr)
        y_trans_method = y_trans_method

        if self.affine_type == "normal":
            tensors = y_trans_method(tensors)
        else:  # 随机变化 放大x y 差距
            x_trans_method = torch.nn.Sequential(*x_trans_arr)
            x_trans_method = x_trans_method
            src = tensors[:, :org_nc]
            tar = tensors[:, org_nc:]
            tar = y_trans_method(tar)
            src = x_trans_method(src)
            tensors = torch.cat([src, tar], dim=1)

        # normalized
        src_img = tensors[:, :self.img_nc]
        src_par = tensors[:, self.img_nc:org_nc]
        tar_img = tensors[:, org_nc:org_nc + self.tar_nc]
        tar_par = tensors[:, org_nc + self.tar_nc:]
        src_img = kornia.color.Normalize(0.5, 0.5)(src_img)
        tar_img = kornia.color.Normalize(0.5, 0.5)(tar_img)
        src = torch.cat([src_img, src_par], dim=1)
        tar = torch.cat([tar_img, tar_par], dim=1)

        if self.ske_to_img:
            tmp = src
            src = tar
            tar = tmp
        return src, tar


    def loadImg(self, line):
        items = line.split('/')
        filename = items[-1]
        inPath1 = line

        if (self.forTrain):
            depthPath1 = os.path.join(os.path.join(self.root, '/train/depth'),filename)
            inPath2 = os.path.join(os.path.join(self.root, '/train/sketch'), filename)
        else:
            depthPath1 = os.path.join(os.path.join(self.root, '/test/depth'), filename)
            inPath2 = os.path.join(os.path.join(self.root, '/test/sketch'), filename)
        src = Image.open(inPath1).convert("RGB")
        tar = Image.open(inPath2)
        # read pic
        src = kornia.image_to_tensor(np.array(src, dtype=float)).float() / 255
        tar = kornia.image_to_tensor(np.array(tar, dtype=float)).float()
        if tar.size(0) == 3:
            tar = kornia.color.RgbToGrayscale()(tar)
        tar = tar / 255

        if self.use_par:
            depth_photo = Image.open(depthPath1).convert("L")
            depth_photo = kornia.image_to_tensor(np.array(depth_photo, dtype=float)).float() / 255
            depth_photo = torch.as_tensor(depth_photo).float()
            src = torch.cat([src, depth_photo])
            tar = torch.cat([tar, depth_photo])

            return torch.cat([src, tar]), filename


