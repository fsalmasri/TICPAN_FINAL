import argparse

parser = argparse.ArgumentParser(description='Thermal Colorizing')
parser.add_argument('--num_epochs', default=30, type=int, help='train epoch number')
# parser.add_argument('--num_epochs', default=500, type=int, help='train epoch number')
parser.add_argument('--GPUid', default='0', type=int, help='GPU ID')
parser.add_argument('--Dataset', default='DS/KAISt-MS-day-3th.pkl', type=str,
                    help='Path to dataset')
# parser.add_argument('--Dataset', default='KAISt-MS-night-test.pkl', type=str,
#                     help='Path to dataset')
# parser.add_argument('--Dataset', default='/home/falmasri/Desktop/Datasets/ULB-VISTI-V2', type=str,
#                     help='Path to dataset')
parser.add_argument('--Folder', default='epochs/KAIST', type=str, help='Saving folder')
parser.add_argument('--lrDecay', type=int, default=200, help='input LR video')
parser.add_argument('--decayType', default='step', help='output SR video')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--threads', type=int, default=5, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lambc', type=int, default=1, help='weight on L1 term in objective')
parser.add_argument('--lamba', type=int, default=0.03, help='weight on L1 term in objective')
parser.add_argument('--lambp', type=int, default=1, help='weight on L1 term in objective')
parser.add_argument('--lambt', type=int, default=1, help='weight on L1 term in objective')

parser.add_argument('--Mode', default='test', type=str, help='train/test')
parser.add_argument('--Model', default=1, type=int, help='1:ULB 2:KAIST')

