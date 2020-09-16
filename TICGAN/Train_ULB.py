import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.utils as utils
import numpy as np
from tqdm import tqdm
from models.TICGAN_Models import *
from Helpers.utilities import FeatureExtractor, normalize_batch, validation_methods
from TICGAN.ULB_VISTI_DataLoader import *


def run(opt, device):
    print('===> Loading datasets')
    training_data_loader = DataLoader(dataset=DatasetFromFolder(opt.Dataset, set='train'), num_workers=opt.threads,
                                      batch_size=opt.batch_size, shuffle=True)
    val_data_loader = DataLoader(dataset=DatasetFromFolder(opt.Dataset, set='valid'), num_workers=opt.threads,
                                 batch_size=1, shuffle=False)


    print('===> Building models')
    net_g = Generator().to(device)
    net_d = Discriminator().to(device)
    weights_init(net_g)
    weights_init(net_d)

    criterionGAN = GANLoss(use_lsgan= False).to(device)
    criterionL1 = nn.L1Loss()
    tvLoss = TVLoss(opt.lambt)
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lrg, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lrd, betas=(0.5, 0.999))
    Fex = FeatureExtractor(torchvision.models.vgg16(pretrained=True)).to(device)

    for epoch in range(opt.num_epochs):
        train_bar = tqdm(training_data_loader, desc='Colorizing thermal')
        running_results = {'batch_sizes': 0, 'loss_d': 0, 'loss_g': 0}

        for data, target in train_bar:
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            real_a, real_b = data.to(device), target.to(device)
            fake_b = net_g(real_a)


            ######################
            # Update D network
            ######################

            optimizer_d.zero_grad()
            pred_fake = net_d(real_a.detach(), fake_b.detach())
            loss_d_fake = criterionGAN(pred_fake, False)

            pred_real = net_d(real_a, real_b)
            loss_d_real = criterionGAN(pred_real, True)

            loss_d = (loss_d_fake + loss_d_real) * 0.5
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()


            ######################
            # Update G network
            ######################

            optimizer_g.zero_grad()
            pred_fake = net_d(real_a, fake_b)

            loss_g_adv = criterionGAN(pred_fake, True) * opt.lamba
            loss_g_content = criterionL1(fake_b, real_b) * opt.lambc
            loss_g_tv = tvLoss(fake_b)

            real_b_Fexs = Fex(normalize_batch(real_b))
            fake_b_Fexs = Fex(normalize_batch(fake_b))
            loss_g_percept = 0
            for r, f in zip(real_b_Fexs, fake_b_Fexs):
                loss_g_content += criterionL1(r, f)

            loss_g = loss_g_adv + loss_g_content + loss_g_tv + loss_g_percept * opt.lambp
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            running_results['loss_d'] += loss_d.item() * batch_size
            running_results['loss_g'] += loss_g.item() * batch_size

            train_bar.set_description(desc='[%d/%d] lrg: %.e, lrd: %.e, loss_d: %.6f , loss_g: %.6f'
                                           % (epoch, opt.num_epochs, opt.lrg, opt.lrd,
                                              running_results['loss_d'] / running_results['batch_sizes'],
                                              running_results['loss_g'] / running_results['batch_sizes']))

        torch.save(net_d, opt.Folder + '/final_d.pt')
        torch.save(net_g, opt.Folder + '/final_g.pt')

        if (epoch + 1) % 1 == 0 and epoch >= 0:
            net_g.eval()
            val_bar = tqdm(val_data_loader)
            valing_results = {'psnr': 0, 'ssim': 0, 'rmse': 0, 'batch_sizes': 0}

            for data, target in val_bar:
                batch_size = data.size(0)
                valing_results['batch_sizes'] += batch_size
                data, target = data.to(device), target.to(device)

                with torch.no_grad():
                    colorized = net_g(data)

                mse, batch_ssim, psnr = validation_methods(colorized, target)
                valing_results['ssim'] += batch_ssim
                valing_results['psnr'] += psnr
                valing_results['rmse'] += np.sqrt(mse)

                val_bar.set_description(
                    desc='[Colorizing Thermal] psnr: %.2f dB ssim: %.2f rmse: %.2f' % (
                             valing_results['psnr'] / valing_results['batch_sizes'],
                             valing_results['ssim'] / valing_results['batch_sizes'],
                             valing_results['rmse'] / valing_results['batch_sizes']))

                if (epoch + 1) % 5 == 0 and epoch > 0:
                    colorized = torch.clamp(colorized, 0, 1)
                    val_images = torch.cat((data.cpu(), target.cpu(), colorized.cpu()), dim=0)
                    image = utils.make_grid(val_images, nrow=3, padding=5)
                    utils.save_image(image, opt.Folder + '/imgs/epoch_%i_%i_.png' % (epoch, valing_results['batch_sizes']),
                                     padding=5)




