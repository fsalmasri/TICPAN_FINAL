import os.path
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
from Helpers.utilities import validation_methods
from Helpers.KAIST_DataLoader import *



def run(opt, device):
    print('===> Loading datasets')
    testing_data_loader = DataLoader(dataset=DatasetFromFolder(opt.Dataset, set='test'), num_workers=opt.threads,
                                     batch_size=1, shuffle=False)

    print('===> Building models')
    net_g = torch.load(opt.Folder + '/final_g.pt', map_location="cuda:%i" % opt.GPUid)
    net_g.eval()

    test_bar = tqdm(testing_data_loader, desc='Colorizing thermal')
    running_results = {'batch_sizes': 0, 'psnr': 0, 'rmse': 0, 'ssim': 0}

    for data, target, fname in test_bar:
        fname = fname
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            colorized = net_g(data)

        mse, batch_ssim, psnr = validation_methods(colorized, target)
        running_results['ssim'] += batch_ssim
        running_results['psnr'] += psnr
        running_results['rmse'] += math.sqrt(mse)

        test_bar.set_description(desc='PSNR: %.6f  SSIM: %.6f   RMSE: %.6f'
                                      % (running_results['psnr'] / running_results['batch_sizes'],
                                         running_results['ssim'] / running_results['batch_sizes'],
                                         running_results['rmse'] / running_results['batch_sizes']))

        dic = opt.Folder + '/test/' + fname[0][0] #.replace('\\', '/')
        if not os.path.exists(dic):
            os.makedirs(dic)

        ndarr = colorized[0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(join(dic, 'C_' + fname[1][0] + '.png'))
