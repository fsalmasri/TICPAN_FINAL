from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
from Helpers.utilities import LoadData, read_files
import torch
from skimage.color import rgb2lab

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


    def __getitem__(self, index):
        RGB_image = Image.open(join(self.RGBdr, self.RGBfiles[index]))
        TI_image = LoadData(join(self.CSVdr, '%i.pkl' %int(self.RGBfiles[index][:-4])))

        LAB_image = rgb2lab(RGB_image)
        LAB_image = normalize_LAB(LAB_image)
        TI_image = ToTensor()(TI_image).float()
        TI_image = torch.cat((TI_image, TI_image, TI_image), dim=0)

        return TI_image, ToTensor()(LAB_image).float(), ToTensor()(RGB_image)

    def __len__(self):
        return len(self.CSVfiles)


def normalize_LAB(img):
    img[:, :, 0] = img[:, :, 0] / 100.0
    img[:, :, 1] = (img[:, :, 1] + 86) / 185
    img[:, :, 2] = (img[:, :, 2] + 107) / 202
    return img

def denormalize_tensorLAB(img):
    L, A, B = img[:,0:1], img[:,1:2], img[:,2:3]
    L = L * 100.0
    A = (A * 185.0) - 86
    B = (B * 202.0) - 107.0
    return torch.cat((L,A,B), dim=1)
