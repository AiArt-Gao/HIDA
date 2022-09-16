import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(Generator, self).__init__()
        #256*256
        self.en1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1),
            #DyReLUB(ngf, conv_type='2d')
            nn.LeakyReLU(0.2, True)
        )
        self.en2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*2),
            #DyReLUB(ngf*2, conv_type='2d')
            nn.LeakyReLU(0.2, True)
        )
        self.en3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            #DyReLUB(ngf*4, conv_type='2d')
            nn.LeakyReLU(0.2, True)
        )
        self.en4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            #DyReLUB(ngf * 8, conv_type='2d')
            nn.LeakyReLU(0.2, True)
        )
        self.en5 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            #DyReLUB(ngf * 8, conv_type='2d')
            nn.LeakyReLU(0.2, True)
        )
        self.en6 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            #DyReLUB(ngf * 8, conv_type='2d')
            nn.LeakyReLU(0.2, True)
        )
        self.en7 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            #DyReLUB(ngf * 8, conv_type='2d')
            nn.LeakyReLU(0.2, True)
        )
        self.en8 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            #DyReLUB(ngf * 8, conv_type='2d')
            nn.ReLU(True)
        )
        self.de1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8,ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            # nn.Dropout(0.5),
            #DyReLUB(ngf * 8, conv_type='2d')
            nn.ReLU(True)
        )
        self.de2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            # nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5),
            #DyReLUB(ngf * 8, conv_type='2d')
            nn.ReLU(True)
        )
        self.de3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            # nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5),
            #DyReLUB(ngf * 8, conv_type='2d')
            nn.ReLU(True)
        )
        self.de4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            # nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5),
            #DyReLUB(ngf * 8, conv_type='2d')
            nn.ReLU(True)
        )
        self.de5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            # nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            #DyReLUB(ngf * 4, conv_type='2d')
            nn.ReLU(True)
        )
        self.de6 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1),
            # nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            #DyReLUB(ngf * 2, conv_type='2d')
            nn.ReLU(True)
        )
        self.de7 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1),
            # nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            #DyReLUB(ngf, conv_type='2d')
            nn.ReLU(True)
        )
        self.de8 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc,kernel_size=4, stride=2,padding=1),
            # nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        #encoder
        out_en1 = self.en1(x)
        #print("out_en1:"+ str(out_en1.shape))
        out_en2 = self.en2(out_en1)
        #print("out_en2:" + str(out_en2.shape))
        out_en3 = self.en3(out_en2)
        #print("out_en3:" + str(out_en3.shape))
        out_en4 = self.en4(out_en3)
        #print("out_en4:" + str(out_en4.shape))
        out_en5 = self.en5(out_en4)
        #print("out_en5:" + str(out_en5.shape))
        out_en6 = self.en6(out_en5)
        #print("out_en6:" + str(out_en6.shape))
        out_en7 = self.en7(out_en6)
        #print("out_en7:" + str(out_en7.shape))
        out_en8 = self.en8(out_en7)
        #print("out_en8:" + str(out_en8.shape))
        #decoder
        out_de1 = self.de1(out_en8)
        #print("out_de1:" + str(out_de1.shape))
        out_de1 = torch.cat((out_de1, out_en7), 1)
        #print("out_de1:" + str(out_de1.shape))
        out_de2 = self.de2(out_de1)
        #print("out_de2:" + str(out_de2.shape))
        out_de2 = torch.cat((out_de2, out_en6), 1)
        #print("out_de2:" + str(out_de2.shape))
        out_de3 = self.de3(out_de2)
        #print("out_de3:" + str(out_de3.shape))
        out_de3 = torch.cat((out_de3, out_en5), 1)
        #print("out_de3:" + str(out_de3.shape))
        out_de4 = self.de4(out_de3)
        #print("out_de4:" + str(out_de4.shape))
        out_de4 = torch.cat((out_de4, out_en4), 1)
        #print("out_de4:" + str(out_de4.shape))
        out_de5 = self.de5(out_de4)
        #print("out_de5:" + str(out_de5.shape))
        out_de5 = torch.cat((out_de5, out_en3), 1)
        #print("out_de5:" + str(out_de5.shape))
        out_de6 = self.de6(out_de5)
        #print("out_de6:" + str(out_de6.shape))
        out_de6 = torch.cat((out_de6, out_en2), 1)
        #print("out_de6:" + str(out_de6.shape))
        out_de7 = self.de7(out_de6)
        #print("out_de7:" + str(out_de7.shape))
        out_de7 = torch.cat((out_de7, out_en1), 1)
        #print("out_de7:" + str(out_de7.shape))
        out_de8 = self.de8(out_de7)
        #print("out_de8:" + str(out_de8.shape))
        # return out_de8
        return out_de7

def unet(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Generator(input_nc=1,output_nc=1)
    if weights_path:
        # state_dict = torch.load(weights_path)
        # model.load_state_dict(state_dict)
        # # # 防止反向传播更新预训练模型参数
        # for p in model.parameters():
        #     p.requires_grad = False  # fune-tuning

        pre_dic = torch.load(weights_path)
        # model.fc_8 = nn.Linear(in_features=4096,  out_features=268, bias=True)
        # model.softmax = nn.Sigmoid()
        model_dict = model.state_dict()
        pre_dic = {k: v for k, v in pre_dic.items() if k in model_dict}
        model_dict.update(pre_dic)
        model.load_state_dict(model_dict)
    else:
        print('weights_path==NULL')

    for p in model.parameters():
        p.requires_grad = False
    return model


if __name__ == '__main__':
    weights_path = 'net_D_ins800.pth'
    unet(weights_path)