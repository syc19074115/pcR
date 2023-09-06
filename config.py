# -- coding: utf-8 --
import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=2,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
#parser.add_argument('--gpu_id', type=list,default=0, help='use cpu only')
parser.add_argument('--gpu_id', type=str,default='cuda:0', help='use cuda')
parser.add_argument('--seed', type=int, default=47, help='random seed')

# data in/out and dataset
parser.add_argument('--dataset_path',default = './dataset',help='fixed trainset root path')
parser.add_argument('--test_data_path',default = './dataset',help='Testset path')
parser.add_argument('--save_path',default='./model/model13',help='save path of trained model')
parser.add_argument('--log_path',default='./runs/log/test13',help='save path of log')
parser.add_argument('--batch_size', type=list, default=16,help='batch size of trainset')  #2

# train
parser.add_argument('--epochs', type=int, default=300, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.0001)')
parser.add_argument('--warmup', type=int, default=30)
parser.add_argument('--resume', default=None, type=str)
#parser.add_argument('--resume', default='./model/model9/model_last.pth', type=str)



args = parser.parse_args()