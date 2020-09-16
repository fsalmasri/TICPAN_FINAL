from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor

from Helpers.utilities import LoadData, read_files
import numpy as np
import torch
import random
from imgaug import augmenters as iaa

class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, set='train'):
        super(DatasetFromFolder, self).__init__()
        self.DS_Directory = dataset_dir
        self.set = set

        if set == 'train':
            self.RGBdr = join(self.DS_Directory, 'Train/CroppedRGB')
            self.CSVdr = join(self.DS_Directory, 'Train/DeSpiked_pkl')
            self.RGBfiles, self.CSVfiles = read_files(self.RGBdr), read_files(self.CSVdr)
        if set == 'valid':
            self.RGBdr = join(self.DS_Directory, 'Validation/CroppedRGB')
            self.CSVdr = join(self.DS_Directory, 'Validation/DeSpiked_pkl')
            self.RGBfiles, self.CSVfiles = read_files(self.RGBdr)[:50], read_files(self.CSVdr)[:50]
        if set == 'test':
            self.RGBdr = join(self.DS_Directory, 'Test/CroppedRGB')
            self.CSVdr = join(self.DS_Directory, 'Test/DeSpiked_pkl')
            self.RGBfiles, self.CSVfiles = read_files(self.RGBdr), read_files(self.CSVdr)
        if set == 'night':
            self.CSVdr = join(self.DS_Directory, 'Test/night/pkl')
            self.CSVfiles = read_files(self.CSVdr)


    def transform(self, image, thermal):
        # Random horizontal flipping
        if random.random() > 0.5:
            image = np.fliplr(image)
            thermal = np.fliplr(thermal)

        # Random vertical flipping
        if random.random() > 0.5:
            image = np.flipud(image)
            thermal = np.flipud(thermal)

        return image, thermal

    def aug_img(self, TI, RGB):
        seq_img = iaa.Sequential([
            iaa.Affine(rotate=(-30, 30), order=1, mode='symmetric', name="MyAffine"),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)
        ])
        seq_img = seq_img.to_deterministic()
        TI = np.expand_dims(TI, axis=2)
        RGB = np.array(RGB)
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

    def __getitem__(self, index):

        if self.set == 'night':
            TI_image = LoadData(join(self.CSVdr, self.CSVfiles[index]))
            TI_image = (TI_image - TI_image.min()) / ( TI_image.max() -  TI_image.min())
            TI_image = ToTensor()(TI_image).float()
            return TI_image, torch.tensor([])

        RGB_image = Image.open(join(self.RGBdr, self.RGBfiles[index]))
        TI_image = LoadData(join(self.CSVdr, '%i.pkl' % int(self.RGBfiles[index][:-4])))
        # TI_image = Normalize_single(TI_image)

        if self.set == 'train':
            TI_image, RGB_image = self.getPatch(TI_image, np.array(RGB_image), 224) #224 #160
            TI_image, RGB_image = self.aug_img(TI_image, np.array(RGB_image))


        return ToTensor()(TI_image.copy()).float(), ToTensor()(RGB_image.copy()).float()

    def __len__(self):
        return len(self.CSVfiles)




# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
#
# def DeNormalize_lab(img):
#     fmean = np.array([[[0, 86.185, 107.863]]])
#     fstd = np.array([[[100, 184.439, 202.345]]])
#
#     return (img * fstd) - fmean
#
# def Normalize_single(x):
#     return (x - x.min()) / (x.max() - x.min())
#
# def Normalize(x):
#     inp = x.view(x.size(0), -1)
#     xMax, _ = inp.max(dim=1)
#     xMin, _ = inp.min(dim=1)
#     xMax = xMax.view(xMax.size(0), 1, 1, 1)
#     xMin = xMin.view(xMin.size(0), 1, 1, 1)
#     xNorm = (x - xMin) / (xMax - xMin)
#     return xNorm.squeeze(0)
#
#
# def normalize_LAB(img):
#     img[:, :, 0] = img[:, :, 0] / 100.0
#     img[:, :, 1] = (img[:, :, 1] + 86) / 185
#     img[:, :, 2] = (img[:, :, 2] + 107) / 202
#     return img
#
# def denormalize_LAB(img):
#     img[:, :, 0] = img[:, :, 0] * 100.0
#     img[:, :, 1] = (img[:, :, 1] * 185) - 86
#     img[:, :, 2] = (img[:, :, 2] * 202) - 107
#     return img
#
# def normalize_tensorLAB(LAB_image):
#     LAB_image[0, :, :] = LAB_image[0, :, :] / 100.0
#     LAB_image[1, :, :] = (LAB_image[1, :, :] + 86) / 185
#     LAB_image[2, :, :] = (LAB_image[2, :, :] + 107) / 202
#     return LAB_image
#
#
# def denormalize_tensorLAB(img):
#     L, A, B = img[:,0:1], img[:,1:2], img[:,2:3]
#     L = L * 100.0
#     A = (A * 185.0) - 86
#     B = (B * 202.0) - 107.0
#     return torch.cat((L,A,B), dim=1)
