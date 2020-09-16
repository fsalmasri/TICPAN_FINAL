from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import torch
from Helpers.utilities import LoadData, read_files


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


    def __getitem__(self, index):

        if self.set == 'night':
            TI_image = LoadData(join(self.CSVdr, self.CSVfiles[index]))
            TI_image = (TI_image - TI_image.min()) / ( TI_image.max() -  TI_image.min())
            TI_image = ToTensor()(TI_image).float()
            return 1 - torch.cat((TI_image,TI_image,TI_image), dim=0), torch.tensor([])

        RGB_image = Image.open(join(self.RGBdr, self.RGBfiles[index]))
        TI_image = LoadData(join(self.CSVdr, '%i.pkl' %int(self.RGBfiles[index][:-4])))
        TI_image = ToTensor()(TI_image).float()
        RGB_image = ToTensor()(RGB_image).float()

        return torch.cat((TI_image,TI_image,TI_image), dim=0), RGB_image

    def __len__(self):
        return len(self.CSVfiles)

