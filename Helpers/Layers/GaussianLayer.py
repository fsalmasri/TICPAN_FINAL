import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn

class GaussianLayer(nn.Module):
    def __init__(self, layers, k=21, sigma=9):
        super(GaussianLayer, self).__init__()

        self.k = k
        self.pad = int((self.k-1)/2) # 10 = (21 - 1 / 2)
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.pad),
            nn.Conv2d(layers, layers, k, stride=1, padding=0, bias=None, groups=layers)
        )

        self.weights_init(sigma)

    def forward(self, x, sigma=0):

        if sigma != 0:
            self.weights_init(sigma)

        return self.seq(x)

    def weights_init(self, sig):
        n= np.zeros((self.k,self.k))
        n[self.pad,self.pad] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=sig)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))


