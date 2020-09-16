import numpy as np
from math import log10
from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys
import os.path
import argparse
import csv
import math
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils
from tqdm import tqdm
import itertools

from models.myGVSIFTI2 import Model
from TICPan.ULB_VISTI_DataLoader import *
from Helpers.Layers.GaussianLayer import GaussianLayer
from Helpers.utilities import SaveData, validation_methods, TVLoss

parser = argparse.ArgumentParser(description='Colorizing ULB using GVSIFTI')
parser.add_argument('--num_epochs', default=1000, type=int, help='train epoch number')
parser.add_argument('--GPUid', default='1', type=int, help='GPU ID')
parser.add_argument('--Dataset', default='/home/falmasri/Desktop/Datasets/ULB-VISTI-V2', type=str, help='Path to dataset')
parser.add_argument('--Mode', default='test', type=str, help='train or test')
parser.add_argument('--Checkpoint', default='default', type=str, help='Path to checkpoint for training')
parser.add_argument('--Folder', default='Feras/AB_HF_ULB_L1_11_ablation6I', type=str, help='Saving folder')
parser.add_argument('--lrDecay', type=int, default=400, help='input LR video')
parser.add_argument('--decayType', default='step', help='output SR video')
parser.add_argument('--lr', type=float, default=8e-4, help='learning rate')
parser.add_argument('--patchSize', type=int, default=32, help='patch size')




##############
def test():
    opt = parser.parse_args()
    G = torch.load('epochs/' + opt.Folder + '/final.pt', map_location="cuda:%i" % opt.GPUid)
    # G = Model().cuda(opt.GPUid)
    # checkpoint = torch.load(os.getcwd() + '/epochs/' + opt.Folder + '/netG_training_epoch_1000.pt')
    # G.load_state_dict(checkpoint['model_state_dict'])

    Gauss3 = GaussianLayer(layers=3, k=25, sigma=12).cuda(opt.GPUid)
    G.eval()

    ds = opt.Dataset
    test_loader = DataLoader(dataset=DatasetFromFolder(ds, set='night'), num_workers=5, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='Colorizing thermal')

    running_results = {'batch_sizes': 0, 'psnr': 0, 'rmse': 0, 'ssim': 0}
    for data, target in test_bar:
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        data = 1 - torch.cat((data, data, data), dim=1)
        data, target = data.cuda(opt.GPUid), target.cuda(opt.GPUid)
        with torch.no_grad():
            data_LF = Gauss3(data)
            data_HF = data - data_LF
            colorized = G(data)
            Gcolorized = Gauss3(colorized)
            colorized_img = Gcolorized + 3 * data_HF
            colorized_img = torch.clamp(colorized_img, 0, 1)

        # mse, batch_ssim, psnr = validation_methods(colorized_img, target)
        # running_results['ssim'] += batch_ssim
        # running_results['psnr'] += psnr
        # running_results['rmse'] += math.sqrt(mse.item())
        #
        # test_bar.set_description(desc='PSNR: %.6f  SSIM: %.6f   RMSE: %.6f'
        #                               % (running_results['psnr'] / running_results['batch_sizes'],
        #                                  running_results['ssim'] / running_results['batch_sizes'],
        #                                  running_results['rmse'] / running_results['batch_sizes']))


        dic = 'epochs/' + opt.Folder + '/night'
        if not os.path.exists(dic):
            os.makedirs(dic)

        ndarr = colorized_img[0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(join(dic, 'C_%i'%running_results['batch_sizes'] + '.png'))

        # target = target[0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()




        # import cv2
        # blured_thermal = cv2.bilateralFilter(colorized2[0].transpose(1,2,0).astype(np.float32), 12, 0.05, 12)

        # plt.subplot(221)
        # plt.imshow(target)
        # plt.subplot(222)
        # plt.imshow(ndarr)
        # # plt.subplot(223)
        # # plt.imshow(blured_thermal)
        # # plt.subplot(224)
        # # plt.imshow(colorized2[0].transpose(1,2,0))
        # plt.show()


        # sys.exit()

####################



def set_lr(args, epoch, optimizer):
    lrDecay = args.lrDecay
    decayType = args.decayType
    if decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2 ** epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'inv':
        k = 1 / lrDecay
        lr = args.lr / (1 + k * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train():

    opt = parser.parse_args()
    G = Model()
    Gauss3 = GaussianLayer(layers=3, k=25, sigma=12).cuda(1)
    print('# generator parameters:', sum(param.numel() for param in G.parameters()))

    G.apply(weights_init)
    l1_loss = nn.L1Loss() # nn.SmoothL1Loss() #
    mse = nn.MSELoss()
    tvLoss = TVLoss(1)
    if torch.cuda.is_available():
        G.cuda(opt.GPUid)
        Gauss3.cuda(opt.GPUid)

    g_optimizer = optim.Adam(G.parameters(), lr=opt.lr)
    totalpsnr = []
    totalLoss = []

    if opt.Checkpoint != 'default':
        checkpoint = torch.load(os.getcwd()+ '/epochs/' + opt.Folder +'/netG_training_epoch_1000.pt')
        G.load_state_dict(checkpoint['model_state_dict'])
        g_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochInit = checkpoint['epoch']
        totalLoss = checkpoint['loss']
        totalpsnr = checkpoint['psnr']
        print('Model Loaded...')
    else :
        epochInit = 0


    ds = opt.Dataset
    train_set = DatasetFromFolder(ds, set='train')
    val_set = DatasetFromFolder(ds, set='valid')
    train_loader = DataLoader(dataset=train_set, num_workers=5, batch_size=opt.patchSize, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=5, batch_size=1, shuffle=False)

    for epoch in range(epochInit, opt.num_epochs, 1):
        train_bar = tqdm(itertools.chain(train_loader, train_loader, train_loader, train_loader,train_loader,
                                         train_loader, train_loader, train_loader, train_loader, train_loader,
                                         train_loader, train_loader, train_loader, train_loader),
                         desc='Colorizing thermal')

        running_results = {'batch_sizes': 0, 'Tloss': 0, 'loss': 0, 'Gloss': 0}

        learning_rate = set_lr(opt, epoch, g_optimizer)
        for data, target in train_bar:
            data = 1 - torch.cat((data, data, data), dim=1)
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            data, target = Variable(data), Variable(target)
            if torch.cuda.is_available():
                data, target = data.cuda(opt.GPUid), target.cuda(opt.GPUid)

            Gtarget = Gauss3(target, 12)

            G.train()
            G.zero_grad()

            colorized_img = G(data)
            Gcolorized_img = Gauss3(colorized_img, 12)
            loss = l1_loss(colorized_img, target)
            Gloss = mse(Gcolorized_img, Gtarget)

            total_loss = loss + 10 * Gloss #+ tvLoss(colorized_img)
            total_loss.backward()
            g_optimizer.step()

            running_results['loss'] += loss.item() * batch_size
            running_results['Gloss'] += Gloss.item() * batch_size
            running_results['Tloss'] += total_loss.item() * batch_size

            train_bar.set_description(desc='[%d/%d] lr: %.e, Loss: %.6f , GLoss: %.6f'
                                           % (epoch, opt.num_epochs, learning_rate,
                                              running_results['loss'] / running_results['batch_sizes'],
                                              running_results['Gloss'] / running_results['batch_sizes']))

        totalLoss.append([running_results['Tloss'] / running_results['batch_sizes']])
        SaveData(totalLoss,'epochs/' + opt.Folder, '/loss.pkl')

        if (epoch + 1)% 1 == 0 and epoch >= 0:
            G.eval()
            val_bar = tqdm(val_loader)

            valing_results = {'psnr_F': 0, 'ssim_F': 0, 'psnr_G': 0, 'ssim_G': 0, 'batch_sizes': 0}
            c=0
            for data, target in val_bar:
                c+=1
                rgb = target.clone()
                batch_size = data.size(0)
                valing_results['batch_sizes'] += batch_size
                data = Variable(data)
                data = 1 - torch.cat((data, data, data), dim=1)
                target = Variable(target)

                if torch.cuda.is_available():
                    data = data.cuda(opt.GPUid)
                    target = target.cuda(opt.GPUid)
                    rgb = rgb.cuda(opt.GPUid)

                with torch.no_grad():
                    target = Gauss3(target)
                    data_LF = Gauss3(data)
                    data_HF = data - data_LF
                    colorized = G(data)
                    Gcolorized = Gauss3(colorized)

                psnr_colorized = Gcolorized + 3 * data_HF

                _, batch_ssim, psnr = validation_methods(Gcolorized, target)
                valing_results['ssim_G'] += batch_ssim
                valing_results['psnr_G'] += psnr

                _, batch_ssim, psnr = validation_methods(torch.clamp(psnr_colorized, 0, 1), rgb)
                valing_results['ssim_F'] += batch_ssim
                valing_results['psnr_F'] += psnr


                val_bar.set_description(
                    desc='[Colorizing Thermal] psnr_F: %.2f dB ssim_F: %.2f '
                         'psnr_G: %.2f dB ssim_G: %.2f '% (
                        valing_results['psnr_F'] / valing_results['batch_sizes'],
                        valing_results['ssim_F'] / valing_results['batch_sizes'],
                        valing_results['psnr_G'] / valing_results['batch_sizes'],
                        valing_results['ssim_G'] / valing_results['batch_sizes']))


                if (epoch + 1) % 50 == 0 and epoch > 0:
                    val_images = []

                    target = target + 3 * data_HF
                    Gcolorized = Gcolorized + 3 * data_HF
                    target = torch.clamp(target, 0, 1)
                    Gcolorized = torch.clamp(Gcolorized, 0, 1)

                    val_images.extend(
                        [data.squeeze().cpu(), rgb.squeeze().cpu(), target.squeeze().cpu(), Gcolorized.squeeze().cpu()])

                    val_images = torch.stack(val_images)
                    val_images = torch.chunk(val_images, val_images.size(0) // 4)

                    dic = 'epochs/' + opt.Folder + '/imgs'
                    if not os.path.exists(dic):
                        os.makedirs(dic)

                    for image in val_images:
                        image = utils.make_grid(image, nrow=4, padding=5)
                        utils.save_image(image, 'epochs/' + opt.Folder + '/imgs/epoch_%i_%i_.png' % (epoch+1, c),
                                         padding=5)

            totalpsnr.append([valing_results['psnr_F'] / valing_results['batch_sizes'],
                              valing_results['ssim_F'] / valing_results['batch_sizes'],
                              valing_results['psnr_G'] / valing_results['batch_sizes'],
                              valing_results['ssim_G'] / valing_results['batch_sizes']])

            SaveData(totalpsnr, 'epochs/' + opt.Folder, '/psnr.pkl')
            torch.save(G, 'epochs/' + opt.Folder + '/final.pt')

            torch.cuda.empty_cache()

        if (epoch+1) % 50 == 0 and epoch > 0:
            torch.save({
                    'epoch': epoch +1 ,
                    'model_state_dict': G.state_dict(),
                    'optimizer_state_dict': g_optimizer.state_dict(),
                    'loss': totalLoss,
                    'psnr': totalpsnr,
                    }, os.getcwd()+ '/epochs/' + opt.Folder + '/netG_training_epoch_%d.pt' %(epoch+1))


#####
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        #nn.init.xavier_normal_(m.weight)
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
#####           

    
###################

if __name__ == "__main__":
    opt = parser.parse_args()
    if opt.Mode == 'test':
        test()
    elif opt.Mode == 'train':
        train()
    sys.exit()
