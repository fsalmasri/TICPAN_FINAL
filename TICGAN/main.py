from TICGAN.Train_KAIST import run as trainKAIST
from TICGAN.Test_KAIST import run as testKAIST
from TICGAN.Train_ULB import run as trainULB
from TICGAN.Test_ULB import run as testULB
from TICGAN.options import *
import torch

args = parser.parse_args()
device = torch.device("cuda:%i" % args.GPUid if True else "cpu")

if __name__ == "__main__":
    if args.Mode == 'test':
        if args.Model == 1: testULB(args, device)
        elif args.Model == 2: testKAIST(args, device)
    elif args.Mode == 'train':
        if args.Model == 1: trainULB(args, device)
        elif args.Model == 2: trainKAIST(args, device)



