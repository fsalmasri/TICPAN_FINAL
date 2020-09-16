import torch.nn as nn
import torch.nn.functional as F
import torch
import sys


def grayBlock(inc, outch, kernel, str, pad):
    return nn.Sequential(
        nn.ReflectionPad2d(pad),
        nn.Conv2d(inc, outch, kernel, str),
        nn.BatchNorm2d(outch),
        nn.LeakyReLU(0.2)
    )


def greenBlock(inc, outch, kernel, str, pad):
    return nn.Sequential(
        nn.ReflectionPad2d(pad),
        nn.Conv2d(inc, outch, kernel, str),
        nn.BatchNorm2d(outch),
        nn.Dropout(0.5)
    )

def upGreenBlock(inc, outch, kernel, str, pad):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        greenBlock(inc, outch, kernel, str, pad)
    )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.b1 = grayBlock(3, 32, 4, 2, 1)
        self.b2 = grayBlock(32, 64, 4, 2, 1)
        self.b3 = grayBlock(64, 128, 3, 1, 1)
        self.b4 = grayBlock(128, 128, 3, 1, 1)

        self.b5 = greenBlock(128, 128, 3, 1, 1)
        self.b6 = greenBlock(256, 128, 3, 1, 1)
        self.b7 = upGreenBlock(192, 64, 3, 1, 1)
        self.b8 = upGreenBlock(96, 32, 3, 1, 1)
        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 3, 3, 1)
        )

        self.relu = nn.ReLU()


    def forward(self, x):

        y1 = self.b1(x)
        y2 = self.b2(y1)
        y3 = self.b3(y2)
        y4 = self.b4(y3)

        inter5 = self.b5(y4)
        y5 = self.relu(torch.cat((inter5, y3), 1))
        inter6 = self.b6(y5)
        y6 = self.relu(torch.cat((inter6, y2), 1))
        inter7 = self.b7(y6)
        y7 = self.relu(torch.cat((inter7, y1), 1))
        y8 = self.b8(y7)
        return self.final(y8)


