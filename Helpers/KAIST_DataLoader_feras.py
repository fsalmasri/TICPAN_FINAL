from os.path import isfile, join
from os import listdir
from PIL import Image
from skimage.color import rgb2lab
from torch.nn.functional import l1_loss,interpolate

from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop , Resize

from Helpers.utilities import LoadData
from torch.nn.functional import interpolate
from imgaug import augmenters as iaa
import torch
import sys
import numpy as np
import random

random.seed(0)


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, set='train'):
        super(DatasetFromFolder, self).__init__()
        self.DS_Directory = '/home/falmasri/Desktop/Datasets/KAIST-MS'
        self.set = set

        if set == 'train':
            [self.TI_filenames  , self.RGB_filenames], _, _ = LoadData(dataset_dir)
        if set == 'valid':
            _, [self.TI_filenames, self.RGB_filenames], _ = LoadData(dataset_dir)
            self.TI_filenames = self.TI_filenames[:50]
            self.RGB_filenames = self.RGB_filenames[:50]
        if set == 'test':
            _, _, [self.TI_filenames  , self.RGB_filenames] = LoadData(dataset_dir)

        self.Trans = tensor_transform()

    def __getitem__(self, index):

        TIPath = join(self.DS_Directory,self.TI_filenames[index].replace('\\','/'))
        RGBPath = join(self.DS_Directory, self.RGB_filenames[index].replace('\\', '/'))

        tensor_scale = Resize((256, 320), interpolation=Image.BICUBIC)
        TI_image = tensor_scale(Image.open(TIPath))
        VIS_img = tensor_scale(Image.open(RGBPath))


        if self.set == 'train':
            TI_image = np.array(TI_image)
            VIS_img = np.array(VIS_img)
            TI_image, VIS_img = self.getPatch(TI_image, VIS_img, 224) #160
            TI_image, VIS_img = self.aug_img(TI_image, VIS_img)


        if self.set =='test':
            # dname = self.TI_filenames[index][9:-16]  # day
            dname = self.TI_filenames[index][11:-16]  # night
            # fname = self.TI_filenames[index][25:-4]  # day
            fname = self.TI_filenames[index][27:-4]  # night

            return ToTensor()(TI_image), self.Trans(VIS_img).float(), [dname, fname]

        return ToTensor()(TI_image.copy()), ToTensor()(VIS_img.copy())

    def __len__(self):
        return len(self.TI_filenames)

    def aug_img(self, TI, RGB):
        seq_img = iaa.Sequential([
            iaa.Affine(rotate=(-30, 30), order=1, mode='symmetric', name="MyAffine"),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)
        ])
        seq_img = seq_img.to_deterministic()
        TI_aug = seq_img.augment_image(TI)
        RGB_aug = seq_img.augment_image(RGB)

        return np.squeeze(TI_aug), RGB_aug

    def getPatch(self,imgIn, imgTar, patchSize):
        (ih, iw, c) = imgTar.shape
        tp = patchSize

        ix = random.randrange(0, iw - tp + 1)
        iy = random.randrange(0, ih - tp + 1)

        imgIn = imgIn[iy:iy + tp, ix:ix + tp]
        imgTar = imgTar[iy:iy + tp, ix:ix + tp, :]
        return imgIn, imgTar


def tensor_transform():
    return Compose([
        ToTensor(),
    ])



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])




