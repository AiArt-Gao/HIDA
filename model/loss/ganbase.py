import torch
import torch.nn as nn
from torch.nn import functional as F, MSELoss

from .focalloss import FocalLoss


class GANLoss(nn.Module):
    def __init__(self, loss_type='original', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.Tensor = tensor
        if loss_type == 'original':
            self.loss = nn.BCELoss()
        elif loss_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif loss_type == 'focal':
            self.loss = FocalLoss()
        else:
            print(
                "Error: type of {} is not defination, please choose 'lsgan、focal、normal' as loss type ".format(
                    loss_type))
            raise KeyError

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label).cuda()
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label).cuda()
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss_ver2(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss_ver2, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode  # original
        self.opt = opt
        if gan_mode == 'ls':
            self.loss_fun = MSELoss()
        elif gan_mode == 'focal':
            self.loss_fun = FocalLoss()
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real, a, b, size):
        if target_is_real:
            # if self.real_label_tensor is None:
            #     print("---------------------")
            self.real_label_tensor = self.Tensor(1).fill_(self.real_label).cuda()
            self.real_label_tensor.requires_grad_(False)
            self.real_label_tensor = self.real_label_tensor.expand_as(input)
            # else:
                #中间[a,b]位置有一个size大小的区域为假，其他区域为真
            if a != -1 and b != -1 and size != 0:
                # print("---------------------")
                # self.real_label_tensor = self.Tensor(1).fill_(self.real_label).cuda()
                # self.real_label_tensor.requires_grad_(False)
                # self.real_label_tensor = self.real_label_tensor.expand_as(input)
                self.real_label_tensor[:,:,a:a+size,b:b+size].fill_(self.fake_label).cuda()
                # print(self.real_label_tensor.shape)

            return self.real_label_tensor
        else:
            # if self.fake_label_tensor is None:
            #     # print("--------")
            self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label).cuda()
            # self.fake_label_tensor.requires_grad_(False)
            self.fake_label_tensor = self.fake_label_tensor.expand_as(input)
            if a != -1 and b != -1 and size != 0:
                self.fake_label_tensor[:, :, a:a + size, b:b + size].fill_(self.real_label).cuda()

            return self.fake_label_tensor

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0).cuda()
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True, a = -1, b = -1,size = 0):
        # cos_sim = F.cosine_similarity(x_parisng,y_parsing, dim=1)
        # cos_sim = torch.abs(cos_sim)
        # cos_sim = torch.unsqueeze(cos_sim, dim=1).mean()
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real, a, b, size)
            # print(target_tensor.shape)
            # print(input.shape)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss_fun(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.max(torch.zeros(input.size(), device=input.device), 1 - input)
                    loss = torch.mean(minval)
                else:
                    minval = torch.max(torch.zeros(input.size(), device=input.device), 1 + input)
                    loss = torch.mean(minval)
                # for
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True, a = -1, b = -1,size = 0):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator, a , b ,size)
                # with panalty:
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


def hinge_panalty(pred_real, pred_fake):
    if isinstance(pred_real, list):
        loss = 0
        for i in range(len(pred_real)):
            if isinstance(pred_real[i], list):
                pred_i_real = pred_real[i][-1]
                pred_i_fake = pred_fake[i][-1]
            loss_tensor = nn.MSELoss()(pred_i_fake.mean(), pred_i_real.mean())
            loss_tensor = loss_tensor * loss_tensor * 0.5
            # with panalty:
            bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
            new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
            loss += new_loss
        return loss / len(pred_real)
