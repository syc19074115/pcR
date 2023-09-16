# -- coding: utf-8 --
import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=2,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
#parser.add_argument('--gpu_id', type=list,default=0, help='use cpu only')
parser.add_argument('--gpu_id', type=str,default='cuda:0', help='use cuda')
parser.add_argument('--seed', type=int, default=3407, help='random seed')

# data in/out and dataset
parser.add_argument('--dataset_path',default = './dataset',help='fixed trainset root path')
parser.add_argument('--test_data_path',default = './dataset',help='Testset path')
parser.add_argument('--save_path',default='./model/model21',help='save path of trained model')
parser.add_argument('--log_path',default='./runs/log/test21',help='save path of log')
parser.add_argument('--batch_size', type=list, default=8,help='batch size of trainset')  #2

# train
parser.add_argument('--model', type=str, default='resDiT', help='model')
parser.add_argument('--epochs', type=int, default=300, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0006, metavar='LR',help='learning rate (default: 0.0001)') #下一次尝试0.
parser.add_argument('--warmup', type=int, default=10)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--lamda', type=float, default=0., metavar='Lamda',help='l1_norm') #下一次尝试0.001
#parser.add_argument('--resume', default='./model/model9/model_last.pth', type=str)



args = parser.parse_args()
""" 
import torch
import unfoldNd
input = torch.rand([1,64,8,8,8])
output = unfoldNd.unfoldNd(input,kernel_size=3,stride=2,padding=1)
b = unfoldNd.unfoldNd(input,kernel_size=1,stride=2,padding=0) #torch.Size([1, 64, 64])
print(b.shape)
c = torch.nn.AvgPool3d(kernel_size=3,stride=2,padding=1)(input)  #torch.Size([1, 64, 4, 4, 4])
c = unfoldNd.unfoldNd(c,kernel_size=3,stride=1,padding=1) #torch.Size([1, 64, 64])
print(c.shape)
c = torch.cat((b,c),dim=1)
#output.shape , 64*8**3//64 , c.shape """