import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
import lpips
from torch import nn
from torch.nn import functional as F

# ref = lpips.im2tensor(lpips.load_image(opt.ref_path))
# pred = Variable(lpips.im2tensor(lpips.load_image(opt.pred_path)), requires_grad=True)

class LpipsLoss(nn.Module):
    def __init__(self, device="cuda"):
        super(LpipsLoss, self).__init__()
        loss_lpips = lpips.LPIPS(net='alex',version='0.1')
        self.device = device
        if(self.device == "cuda"):
            self.loss_lpips = loss_lpips.cuda()

    def forward(self, src, tar):
        # assert src.size() == tar.size()
        if(self.device == "cuda"):
            src = src.cuda()
            tar = tar.cuda()
        dist = self.loss_lpips.forward(tar, src)
        return dist


if __name__ == '__main__':
    lpips = LpipsLoss(device="cuda")
    img1 = torch.randn([1, 3, 256, 256])
    img2 = torch.randn([1, 3, 256, 256])
    g = lpips.forward(img1, img2)
    print(g)
