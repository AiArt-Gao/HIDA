import functools
import torch
from torch import nn
from config.SAND_pix_opt import TrainOptions
from model.base_network import BaseNetwork
from torch.nn import functional as F
import numpy as np
from torch.nn import utils


class MultiscaleDiscriminator(BaseNetwork):

    def __init__(self, opt, in_c):
        super().__init__()
        self.opt = opt
        self.in_c = in_c
        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt, self.in_c)
        elif subarch == 'n_layer_style':
            netD = NLayerDiscriminator_style(opt, self.in_c)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []

        for name, D in self.named_children():
            out = D(input)
            result.append(out)
            input = self.downsample(input)

        return result

class MultiscaleDiscriminator_SN(BaseNetwork):

    def __init__(self, opt, in_c):
        super().__init__()
        self.opt = opt
        self.in_c = in_c
        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator_SN(opt, self.in_c)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []

        for name, D in self.named_children():
            out = D(input)
            result.append(out)
            input = self.downsample(input)

        return result


class CycleGANDiscriminator(BaseNetwork):
    def __init__(self, opt, in_c):
        super(CycleGANDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(in_c, opt.ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, opt.n_layers_D):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(opt.ndf * nf_mult_prev, opt.ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(opt.ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** opt.n_layers_D, 8)
        sequence += [
            nn.Conv2d(opt.ndf * nf_mult_prev, opt.ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(opt.ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(opt.ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class NLayerDiscriminator(BaseNetwork):

    def __init__(self, opt, in_c, only_last=False):
        super().__init__()
        self.opt = opt
        self.only_last = only_last
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = in_c
        norm_layer = torch.nn.InstanceNorm2d
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]
        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [
                [norm_layer(nf_prev), nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw),
                 nn.LeakyReLU(0.2, False)
                 ]]

        sequence += [[nn.InstanceNorm2d(512, affine=False, track_running_stats=False),
                      nn.LeakyReLU(0.2, False)]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        input_nc = opt.image_nc + opt.parsing_nc + opt.output_nc
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)
        if self.only_last:
            return results[-1]
        else:
            return results[1:]


class NLayerDiscriminator_style(BaseNetwork):
#x增加了style的
    def __init__(self, opt, in_c, only_last=False):
        super().__init__()
        self.opt = opt
        self.only_last = only_last
        self.kw = 4
        self.padw = int(np.ceil((self.kw - 1.0) / 2))
        self.nf = opt.ndf
        input_nc = in_c
        norm_layer = torch.nn.InstanceNorm2d
        sequence = [[nn.Conv2d(input_nc, self.nf, kernel_size=self.kw, stride=2, padding=self.padw),
                     nn.LeakyReLU(0.2, False)]]
        for n in range(1, opt.n_layers_D):
            nf_prev = self.nf
            self.nf = min(self.nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [
                [norm_layer(nf_prev), nn.Conv2d(nf_prev, self.nf, kernel_size=self.kw, stride=stride, padding=self.padw),
                 nn.LeakyReLU(0.2, False)
                 ]]

        sequence += [[nn.InstanceNorm2d(512, affine=False, track_running_stats=False),
                      nn.LeakyReLU(0.2, False)]]

        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        input_nc = opt.image_nc + opt.parsing_nc + opt.output_nc
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        out = results[-1]
        s1 = nn.Conv2d(self.nf, 1, kernel_size=self.kw, stride=1, padding=self.padw).cuda()
        s2 = nn.Conv2d(self.nf, 3, kernel_size=self.kw, stride=1, padding=self.padw).cuda()
        realOrFake = s1(out)#[8,1,35,35]
        styleLabel = s2(out)#[8,3,35,35]
        results.append(realOrFake)
        results.append(styleLabel)

        if self.only_last:
            return results[-1]
        else:
            return results[1:]

class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf=64):
        super(Discriminator, self).__init__()
        self.cov1 = nn.Sequential(
            nn.Conv2d(input_nc + output_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.cov2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )
        self.cov3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True)
        )
        self.cov4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.cov5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out_cov1 = self.cov1(x)
        out_cov2 = self.cov2(out_cov1)
        out_cov3 = self.cov3(out_cov2)
        out_cov4 = self.cov4(out_cov3)
        out = self.cov5(out_cov4)
        return out


if __name__ == '__main__':
    opt = TrainOptions().parse()
    # opt = SPADEGenerator.modify_commandline_options(opt)
    style = torch.randn([2, 12, 256, 256])
    model = NLayerDiscriminator(opt, in_c=12)
    hat_y = model.forward(style)
    print(hat_y[:][-1])
