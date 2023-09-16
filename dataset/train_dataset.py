import torchvision.transforms
from torch.utils.data import DataLoader
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
from .transforms import Rot90, Flip, Identity, Compose, RandCrop3D, RandomShift, RandomRotion ,CenterCrop,RandomFlip
#from transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
#args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'

class Train_Dataset(Dataset):
    def __init__(self,args):
        #super.__init__(self,args)
        self.args = args

        # self.filename_list = self.load_file_name_list('./train_path_list_cut.txt')#test
        #self.filename_list_cut = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list_cut.txt'))
        #self.filename_list_region = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list_region.txt'))
        self.filename_list_region = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))
        #self.Centercrop = CenterCrop([36,70,70])
        self.transforms = Compose([
            #RandomShift(2),
            RandCrop3D((64,64,64)),
            RandomShift(2),
            RandomRotion(2),
            #RandomIntensityChange((0.1,0.1)),
            #RandomFlip(0)
        ])  # 数据增强就体现在这里


    def __getitem__(self, index):
        #pre_cut_ct_array, pre_cut_seg_array, post_cut_ct_array, post_cut_seg_array, label_cut = self.load_cut_or_region_array(self.filename_list_cut,index)
        pre_region_ct_array, pre_region_seg_array, post_region_ct_array, post_region_seg_array, label_region = self.load_cut_or_region_array(self.filename_list_region,index)
        # print(label_region,label_cut)
        #assert label_cut == label_region , "region和cut输出的label不一样"
        if self.transforms:
            #pre_region_ct_array,pre_region_seg_array,post_region_ct_array,post_region_seg_array = self.Centercrop([pre_region_ct_array,pre_region_seg_array,post_region_ct_array,post_region_seg_array])
            pre_region_ct_array,pre_region_seg_array,post_region_ct_array,post_region_seg_array = self.transforms([pre_region_ct_array,pre_region_seg_array,post_region_ct_array,post_region_seg_array])
        #这个是双通道的版本
        pre_array = np.vstack((pre_region_ct_array, pre_region_seg_array))
        #print(pre_array.shape) (2,128,64,64)
        choice = np.random.choice([True,False])
        pre_array = self.sample(pre_array,2,choice)
        post_array = np.vstack((post_region_ct_array, post_region_seg_array))
        post_array = self.sample(post_array,2,choice)
        pre_array = torch.FloatTensor(pre_array.copy())
        post_array = torch.FloatTensor(post_array.copy())
        return pre_array, post_array, label_region

    def __len__(self):
        return len(self.filename_list_region)

    def load_cut_or_region_array(self, filename_list,index):
        ct_pre = sitk.ReadImage(filename_list[index][0].replace('\\', '/').replace('cut','region'), sitk.sitkFloat32)
        seg_pre = sitk.ReadImage(filename_list[index][1].replace('\\', '/').replace('cut','region'), sitk.sitkUInt8)
        ct_post = sitk.ReadImage(filename_list[index][2].replace('\\', '/').replace('cut','region'), sitk.sitkFloat32)
        seg_post = sitk.ReadImage(filename_list[index][3].replace('\\', '/').replace('cut','region'), sitk.sitkUInt8)
        
        label_pcr = np.array(int(filename_list[index][4]))
        # print(type(label_pcr))  #<class 'str'>

        pre_ct_array = sitk.GetArrayFromImage(ct_pre)
        pre_seg_array = sitk.GetArrayFromImage(seg_pre)
        post_ct_array = sitk.GetArrayFromImage(ct_post)
        post_seg_array = sitk.GetArrayFromImage(seg_post)
        # print("pre.shape:{}   post.shape:{}".format(pre_ct_array.shape,post_ct_array.shape)) #pre.shape:(128, 64, 64)   post.shape:(128, 64, 64)

        # print("beigin:",pre_ct_array.shape,post_ct_array.shape)  #(180, 256, 256) (180, 256, 256)
        pre_ct_array = pre_ct_array[None, ...]
        pre_seg_array = pre_seg_array[None, ...]
        post_ct_array = post_ct_array[None, ...]
        post_seg_array = post_seg_array[None, ...]
        # pre_ct_array = torch.FloatTensor(pre_ct_array).unsqueeze(0)  # 在第0维加入一个维度
        # post_ct_array = torch.FloatTensor(post_ct_array).unsqueeze(0)  # torch.Size([1, 180, 256, 256]) torch.Size([1, 180, 256, 256]
        label_pcr = torch.from_numpy(label_pcr).type(torch.LongTensor)
        return pre_ct_array,pre_seg_array,post_ct_array,post_seg_array,label_pcr

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据,strip()函数去掉了行末的换行
                if not lines:
                    break
                file_name_list.append(lines.split())  #split()将lines分成两个元素，然后组成一个list

        # print(file_name_list[0])
        return file_name_list

    def normalize(self,image, chestwl=30.0, chestww=350.0):
        # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        # print(chestwl,chestww)
        Low = chestwl - 0.5 * chestww
        High = chestwl + 0.5 * chestww
        image[image > High] = High
        image[image < Low] = Low
        image = (image - Low) / (High - Low)
        return image

    def normalization(self, data):
        mn = data.mean()
        sd = data.std()
        new_data = (data - mn) / sd
        return new_data

    def sample(self, Array, Magnification = 2, choice = True): #input = (2 , 128 , 64 , 64)
        L = len(Array)
        if choice:
            new_array = Array[:,::Magnification,:,:]
        else:
            new_array = Array[:,1::Magnification,:,:]
        return new_array
    
if __name__ == '__main__':
    sys.path.append('./')
    from config import args
    # print(os.getcwd())
    train_ds = Train_Dataset(args)
    train_dl = DataLoader(train_ds, 1, False, num_workers=1)
    for i, (pre, post, label) in enumerate(train_dl):
        print("--------------------------------{}-------------------------------".format(i))
        print(pre.size(), post.size(),label.size())  # torch.Size([1, 1, 120, 60, 60]) torch.Size([1, 1, 120, 60, 60]) torch.Size([1])
        print(label,type(label)) # tensor([1], dtype=torch.int32) <class 'torch.Tensor'>
        labels_MP = label.view(-1)
        print(labels_MP)
        if i != -1:
            pre = pre.numpy()
            post = post.numpy()
            plt.subplot(141)
            plt.axis('off')
            plt.imshow(pre[0][0][25],cmap='gray')
            plt.subplot(142)
            plt.axis('off')
            plt.imshow(pre[0][1][25],cmap='gray')
            plt.subplot(143)
            plt.axis('off')
            plt.imshow(post[0][0][25],cmap='gray')
            plt.subplot(144)
            plt.axis('off')
            plt.imshow(post[0][1][25],cmap='gray')
            #plt.savefig('./dataset/test_train.jpg')
            plt.savefig('./dataset/test_train/test_transforms-{}.jpg'.format(i))

