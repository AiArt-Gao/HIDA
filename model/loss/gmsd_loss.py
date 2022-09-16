import torch
from torch import nn
from torch.nn import functional as F

'''
    GMSD 使用一个卷积操作，只是卷积采用固定的3x3 prewitt 算子做weight，
'''


class GMSDLoss(nn.Module):
    def __init__(self, in_channel, mid_channel=None, device="cuda", criteration=nn.L1Loss):
        super(GMSDLoss, self).__init__()
        if mid_channel == None:
            mid_channel = in_channel
        self.prewitt_x = torch.FloatTensor([[1. / 3, 0, -1. / 3]]).reshape((1, 1, 1, 3)).to(device)
        self.prewitt_x = self.prewitt_x.expand((in_channel, mid_channel, 3, 3))
        self.prewitt_y = self.prewitt_x.transpose(2, 3)
        self.avg_filter = torch.FloatTensor([[0.25, 0.25], [0.25, 0.25]]).reshape((1, 1, 2, 2)).to(device)
        self.avg_filter = self.avg_filter.expand((in_channel, mid_channel, 2, 2))
        self.criteration = criteration()  # 默认为均方根误差

    def forward(self, src, tar):
        assert src.size() == tar.size()
        mr_sq_x = F.conv2d(src, self.prewitt_x, stride=1, padding=1)
        mr_sq_y = F.conv2d(src, self.prewitt_y, stride=1, padding=1)
        md_sq_x = F.conv2d(tar, self.prewitt_x, stride=1, padding=1)
        md_sq_y = F.conv2d(tar, self.prewitt_y, stride=1, padding=1)
        eps = 1e-7
        frac1 = mr_sq_x.mul(md_sq_x) + mr_sq_y.mul(md_sq_y)
        frac2 = ((mr_sq_y ** 2 + mr_sq_x ** 2 + eps).sqrt()).mul((md_sq_y ** 2 + md_sq_x ** 2 + eps).sqrt())
        gmsd = (1 - frac1 / frac2).mean()
        return gmsd


if __name__ == '__main__':
    gmsd = GMSDLoss(3, device="cpu")
    img1 = torch.randn([1, 3, 256, 256])
    img2 = torch.randn([1, 3, 256, 256])
    g = gmsd.forward(img1, img2)
    print(g)
