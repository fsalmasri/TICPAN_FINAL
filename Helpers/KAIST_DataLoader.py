from os.path import isfile, join
from os import listdir
from PIL import Image
from skimage.color import rgb2lab
from torch.nn.functional import l1_loss,interpolate

from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop , Resize

from Helpers.utilities import LoadData
from imgaug import augmenters as iaa
import torch
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


    def __getitem__(self, index):

        TIPath = join(self.DS_Directory,self.TI_filenames[index].replace('\\','/'))
        RGBPath = join(self.DS_Directory, self.RGB_filenames[index].replace('\\', '/'))

        tensor_scale = Resize((256, 320), interpolation=Image.BICUBIC)
        TI_image = Image.open(TIPath)
        lr_TI_image = tensor_scale(TI_image)

        VIS_img = Image.open(RGBPath)
        lr_VIS = tensor_scale(VIS_img)

        LAB_image = self.Trans(rgb2lab(lr_VIS)).float()
        LAB_image = normalize_tensorLAB(LAB_image)


        if self.set =='test':
            dname = self.TI_filenames[index][9:-16] # day
            # dname = self.TI_filenames[index][11:-16]  # night
            fname = self.TI_filenames[index][25:-4] #day
            # fname = self.TI_filenames[index][27:-4]  # night

            return ToTensor()(lr_TI_image), LAB_image, ToTensor()(lr_VIS).float(), [dname, fname]

        return ToTensor()(lr_TI_image), LAB_image, ToTensor()(lr_VIS).float()

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



def normalize_tensorLAB(LAB_image):
    LAB_image[0, :, :] = LAB_image[0, :, :] / 100.0
    LAB_image[1, :, :] = (LAB_image[1, :, :] + 86) / 185
    LAB_image[2, :, :] = (LAB_image[2, :, :] + 107) / 202
    return LAB_image

def denormalize_LAB(img):
    img[:, :, 0] = img[:, :, 0] * 100.0
    img[:, :, 1] = (img[:, :, 1] * 185) - 86
    img[:, :, 2] = (img[:, :, 2] * 202) - 107
    return img

def denormalize_tensorLAB(img):
    std = torch.Tensor([100.0, 185.0, 202.0]).view(1,3,1,1).to(img)
    mean = torch.Tensor([0.0, 86.0, 107.0]).view(1,3,1,1).to(img)
    img = (img * std) - mean
    return img

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])




