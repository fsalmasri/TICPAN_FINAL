from TIR2LAB.Train_Berge_ULB import train as trainULB
from TIR2LAB.Test_Berge_ULB import test as testULB
from TIR2LAB.Train_Berge_KAIST import train as trainKAIST
from TIR2LAB.Test_Berge_KAIST import test as testKAIST
from TIR2LAB.options import *
import torch

args = parser.parse_args()
device = torch.device("cuda:%i" % args.GPUid if args.use_gpu else "cpu")

if __name__ == "__main__":
    if args.Mode == 'test':
        if args.Model == 1: testULB(args, device)
        elif args.Model == 2: testKAIST(args, device)
    elif args.Mode == 'train':
        if args.Model == 1: trainULB(args, device)
        elif args.Model == 2: trainKAIST(args, device)


