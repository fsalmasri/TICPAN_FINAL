import pickle as cPickle
import os
from os import listdir
from os.path import isfile, join
import torch.nn as nn
from Helpers.Loss import pytorch_ssim
import math
from torch.nn import init
import torch



# def read_images(folder, filesList, requiredSize = None):
#     Imgs = []
#
#     for idx in range(len(filesList)):
#         jpgfile = read_img(join(folder, filesList[idx].replace('\\','/')))
#         if requiredSize != None:
#             # jpgfile = resizeimage.resize_cover(jpgfile, requiredSize)
#             jpgfile = cv2.resize(jpgfile, requiredSize, interpolation=cv2.INTER_AREA)
#         Imgs.append(jpgfile)
#
#     return np.array(Imgs)

# def read_img(file):
#     return np.array(Image.open(file))

def read_files(folderPath):
    onlyfiles = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]

    return onlyfiles

def LoadData(param_toRead):
    if (os.path.isfile(param_toRead)):
        f = open(param_toRead, 'rb')
        data =  cPickle.load(f)
        f.close()
        return data
    else:
        print('err')
        return None

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

def validation_methods(sr, hr):
    batch_mse = ((sr - hr) ** 2).data.mean()
    batch_ssim = pytorch_ssim.ssim(sr, hr).item()
    psnr = 10 * math.log10(1 / (batch_mse / sr.size(0)))

    return batch_mse, batch_ssim, psnr



class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size


def weights_init(net, he=True):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if he:
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if not he:
                init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class FeatureExtractor(nn.Module):
    def __init__(self, cnn):
        super(FeatureExtractor, self).__init__()

        self.features3 = nn.Sequential(*list(cnn.features)[:3])
        self.features8 = nn.Sequential(*list(cnn.features)[:8])
        self.features15 = nn.Sequential(*list(cnn.features)[:15])
        self.features22 = nn.Sequential(*list(cnn.features)[:22])

    def forward(self, x):
        return [self.features3(x), self.features8(x), self.features15(x), self.features22(x)]