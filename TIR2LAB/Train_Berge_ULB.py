import os.path
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.color import lab2rgb
import itertools
import numpy as np
from models.TIR2LAB_model import Model
from Helpers.utilities import validation_methods, weights_init
from TIR2LAB.ULB_VISTI_DataLoader import *
from Helpers.pytorch_ssim import SSIM
from torch.nn.functional import l1_loss
import torchvision.utils as utils
import warnings
warnings.filterwarnings("ignore")



def train(args, device):

    G = Model().to(device)
    weights_init(G, he=False)
    criterion = SSIM(window_size = [4,16])
    optimizer = optim.Adam(G.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(dataset=DatasetFromFolder(args.ULB_Dataset, set='train'), num_workers=5, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=DatasetFromFolder(args.ULB_Dataset, set='valid'), num_workers=5, batch_size=1, shuffle=False)

    for epoch in range(0, args.num_epochs, 1):
        train_bar = tqdm(itertools.chain(train_loader, train_loader, train_loader, train_loader, train_loader,
                                         train_loader, train_loader, train_loader, train_loader, train_loader,
                                         train_loader, train_loader, train_loader, train_loader),
                         desc='Colorizing thermal')

        running_results = {'batch_sizes': 0, 'loss': 0, 'L1loss': 0, 'SSIMloss': 0}
        G.train()
        for data, target, _ in train_bar:
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            data, target = data.to(device), target.to(device)

            G.zero_grad()
            colorized = G(data)

            loss1 = criterion(colorized[:,0:1], target[:,0:1])
            loss2 =  l1_loss(colorized[:, 1:3], target[:, 1:3])
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_results['loss'] += loss.item() * batch_size
            running_results['L1loss'] += loss2.item() * batch_size
            running_results['SSIMloss'] += loss1.item() * batch_size
            train_bar.set_description(desc='[%d/%d] L1_Loss: %.6f  SSIM: %.6f'
                                           %(epoch, args.num_epochs,
                                             running_results['L1loss'] / running_results['batch_sizes'],
                                             running_results['SSIMloss'] / running_results['batch_sizes']))

        torch.save(G, args.Folder + '/final.pt')

        if (epoch + 1)% 1 == 0 and epoch >= 0:
            G.eval()
            val_bar = tqdm(val_loader)

            valing_results = {'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            c=0
            for data, _, target in val_bar:
                c+=1
                batch_size = data.size(0)
                valing_results['batch_sizes'] += batch_size
                data = data.cuda(args.GPUid)

                with torch.no_grad():
                    colorized = G(data)
                    colorized = denormalize_tensorLAB(colorized)
                    colorized = lab2rgb(colorized[0].permute(1, 2, 0).data.cpu().numpy())[np.newaxis,:]
                    colorized = torch.FloatTensor(colorized).permute(0,3,1,2)
                    colorized = torch.clamp(colorized, 0, 1)

                _, batch_ssim, psnr = validation_methods(colorized, target)
                valing_results['ssim'] += batch_ssim
                valing_results['psnr'] += psnr

                val_bar.set_description(
                    desc='[Colorizing Thermal] psnr: %.2f dB ssim: %.2f '
                         % (valing_results['psnr'] / valing_results['batch_sizes'],
                            valing_results['ssim'] / valing_results['batch_sizes']))


                if (epoch + 1) % 10 == 0 and epoch > 0:
                    val_images = []
                    val_images.extend(
                        [data.squeeze().cpu(), target.squeeze().cpu(), colorized.squeeze().cpu()])

                    val_images = torch.stack(val_images)
                    val_images = torch.chunk(val_images, val_images.size(0) // 3)

                    dic = args.Folder + '/imgs'
                    if not os.path.exists(dic):
                        os.makedirs(dic)

                    for image in val_images:
                        image = utils.make_grid(image, nrow=3, padding=5)
                        utils.save_image(image, args.Folder + '/imgs/epoch_%i_%i_.png' % (epoch+1, c),
                                         padding=5)


