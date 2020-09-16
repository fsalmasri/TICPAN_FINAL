import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import lr_scheduler
from torch.nn import init

def convBlock(inc, outch, kernel_size, str=1):
    pad = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.ReflectionPad2d(pad),
        nn.Conv2d(inc, outch, kernel_size, stride= str, padding=0, bias=False),
        nn.InstanceNorm2d(outch, affine=True),
        nn.ReLU()
    )

class Generator(nn.Module):
    def __init__(self):

        super(Generator, self).__init__()

        self.nonres1 = nn.Sequential(
            convBlock(3, 64, 7),
            convBlock(64, 64, 3),
            convBlock(64, 128, 3),
            convBlock(128, 256, 3)
        )

        block_res1 = [ResidualBlock(256) for _ in range(9)]
        self.block_res_n = nn.Sequential(*block_res1)

        self.nonres2 = nn.Sequential(
            convBlock(256, 256, 3),
            convBlock(256, 128, 3),
            convBlock(128, 64, 3)
        )

        self.nonres3 = nn.Sequential(
            convBlock(3, 32, 7),
            convBlock(32, 64, 3)
        )

        block_res2 = [ResidualBlock(64) for _ in range(3)]
        self.block_res_m = nn.Sequential(*block_res2)

        self.nonres4 = nn.Sequential(
            convBlock(64, 32, 3),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 3, 3, stride=1, padding=0, bias=False)
        )

    def forward(self, x):

        nres1 = self.nonres1(x)
        res1 = self.block_res_n(nres1)
        nres2 = self.nonres2(res1)

        nres3 = self.nonres3(x)
        c = nres2 + nres3
        res2 = self.block_res_m(c)
        nres4 = self.nonres4(res2)

        return (torch.tanh(nres4) + 1) / 2


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(channels, affine=True)


    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.nonres1 = nn.Sequential(
            nn.Conv2d(6, 32, 4, stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x, y):

        inp = torch.cat((x,y) , 1)
        return self.nonres1(inp)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, inputx, target_is_real):
        target_tensor = self.get_target_tensor(inputx, target_is_real)
        return self.loss(inputx, target_tensor)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.num_epochs - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def weights_init(net):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    # print('learning rate = %.7f' % lr)
    return lr

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
