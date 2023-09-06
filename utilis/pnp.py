# -- coding: utf-8 --
#即插即用模块

import torch.nn as nn
import torch
class SEModule_3D(nn.Module):  #3d SE
    def __init__(self, in_channels=3, reduction_ratio=16):
        super(SEModule_3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        nn.init.xavier_normal_(self.fc2.weight)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ ,_= x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1,1)
        return x * y

class SEModule_2D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEModule_2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

if __name__ == '__main__':
    net = SEModule_3D(in_channels=3,reduction_ratio=2)
    input = torch.rand([1,3,128,64,64])
    # output = nn.AdaptiveAvgPool3d(output_size=1)(input)
    output = net(input)
    print(output.shape)