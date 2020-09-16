import argparse

parser = argparse.ArgumentParser(description='Thermal Colorization')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--learning_rate', default=0.001, type=int, help='Learning rate')
parser.add_argument('--use_gpu', default=True, help='Train model on GPU')
parser.add_argument('--GPUid', default='1', type=int, help='GPU ID')
parser.add_argument('--Mode', default='test', type=str, help='train/test')
parser.add_argument('--ULB_Dataset', default='Datasets/ULB-VISTI-V2', type=str, help='Path to dataset')
parser.add_argument('--KAIST_Dataset', default='dataset/Timg/KAISt-MS-day-3th.pkl', type=str, help='Path to dataset') #KAISt-MS-night-test
parser.add_argument('--Folder', default='epochs/berge_ULB', type=str, help='Saving folder')
# parser.add_argument('--Folder', default='berge_KAIST', type=str, help='Saving folder')
parser.add_argument('--Model', default=1, type=int, help='1:ULB 2:KAIST')

