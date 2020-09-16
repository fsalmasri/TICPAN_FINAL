import torch.nn as nn
import torch.nn.functional as F
import torch
import sys


def norm(ch):
    # return nn.InstanceNorm2d(ch, affine=True)
    return nn.BatchNorm2d(ch)

def grayBlock(inc, outch, kernel, str):
    padding = kernel - str
    paddingTop = padding // 2
    return nn.Sequential(
        nn.LeakyReLU(0.2),
        nn.ReflectionPad2d((paddingTop, padding - paddingTop, paddingTop, padding - paddingTop)),
        nn.Conv2d(inc, outch, kernel, str),
        norm(outch)
    )


def greenBlock(inc, outch, kernel, str):
    padding = kernel - str
    paddingTop = padding // 2
    return nn.Sequential(
        nn.LeakyReLU(0.2),
        nn.ReflectionPad2d((paddingTop, padding - paddingTop, paddingTop, padding - paddingTop)),
        nn.Conv2d(inc, outch, kernel, str),
        norm(outch),
        nn.Dropout(0.5)
    )


def greenBlock2(inc, outch, kernel, str):
    padding = kernel - str
    paddingTop = padding // 2
    return nn.Sequential(
        nn.LeakyReLU(0.2),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((paddingTop, padding - paddingTop, paddingTop, padding - paddingTop)),
        nn.Conv2d(inc, outch, kernel, str),
        norm(outch),
        nn.Dropout(0.5)
    )


def final(inc, outch, kernel, str):
    padding = kernel - str
    paddingTop = padding // 2
    return nn.Sequential(
        nn.ReflectionPad2d((paddingTop, padding - paddingTop, paddingTop, padding - paddingTop)),
        nn.Conv2d(inc, outch, kernel, str)
    )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.b0 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32, 4, 2)
        )

        self.b1 = grayBlock(32, 64, 4, 2)
        self.b2 = grayBlock(64, 128, 4, 1)
        self.b3 = grayBlock(128, 128, 4, 1)

        self.b4 = greenBlock(128, 128, 4, 1)
        self.b5 = greenBlock(256, 128, 4, 1)

        self.b6 = greenBlock2(192, 64, 4, 1)
        self.b7 = greenBlock2(96, 32, 4, 1)

        self.final = final(32, 3, 4, 1)

    def forward(self, x):
        y0 = self.b0(x)
        y1 = self.b1(y0)
        y2 = self.b2(y1)
        y3 = self.b3(y2)
        y4 = self.b4(y3)

        inter5 = torch.cat((y4, y2), 1)
        y5 = self.b5(inter5)
        inter6 = torch.cat((y5, y1), 1)
        y6 = self.b6(inter6)
        inter7 = torch.cat((y6, y0), 1)
        y7 = self.b7(inter7)
        return self.final(y7)
