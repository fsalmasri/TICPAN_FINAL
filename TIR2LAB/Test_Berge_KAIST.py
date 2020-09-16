import os.path
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.color import lab2rgb
from Helpers.utilities import validation_methods
from Helpers.KAIST_DataLoader import *
import math
import warnings
warnings.filterwarnings("ignore")




def test(args, device):
    G = torch.load('epochs/' + args.Folder + '/final.pt', map_location="cuda:%i" % args.GPUid)
    G.eval()

    test_loader = DataLoader(dataset=DatasetFromFolder(args.Dataset, set='test'),
                             num_workers=5, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='Colorizing thermal')

    running_results = {'batch_sizes': 0, 'psnr': 0, 'rmse': 0, 'ssim': 0}
    for data, _, RGB, fname in test_bar:
        fname = fname
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        data = data.to(device)
        with torch.no_grad():
            colorized_img = G(data)
            colorized_img = denormalize_tensorLAB(colorized_img.data.cpu())
            colorized_img = lab2rgb(colorized_img[0].permute(1, 2, 0).numpy())

            colorized_img = np.clip(colorized_img, 0, 1)
            colorized_img = torch.FloatTensor(colorized_img).permute(2, 0, 1).unsqueeze(0)


        mse, batch_ssim, psnr = validation_methods(colorized_img, RGB)
        running_results['ssim'] += batch_ssim
        running_results['psnr'] += psnr
        running_results['rmse'] += math.sqrt(mse.item())

        test_bar.set_description(desc='PSNR: %.6f  SSIM: %.6f   RMSE: %.6f'
                                      % (running_results['psnr'] / running_results['batch_sizes'],
                                         running_results['ssim'] / running_results['batch_sizes'],
                                         running_results['rmse'] / running_results['batch_sizes']))

        dic = args.Folder + '/test/' + fname[0][0].replace('\\','/')
        if not os.path.exists(dic):
            os.makedirs(dic)

        ndarr = colorized_img[0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(join(dic, 'C_' + fname[1][0] + '.png'))

