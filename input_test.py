# -- coding: utf-8 --
from torch.utils.tensorboard import SummaryWriter
from DIT import DiT_basic
from dataset import train_dataset, val_dataset
import config
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:32'
import numpy as np
from torch.utils.data import DataLoader
import torch, gc
from tqdm import tqdm
import random
from utilis.scheduler import CosineAnnealingLRWarmup
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from dataset.transforms import Rot90, Flip, Identity, Compose
from dataset.transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange

def main():
    path2 = 'dataset/pre_cut/ct/volume-7.nii'
    path3 = 'dataset/pre_cut/seg/segmentation-7.nii'
    path = 'dataset/pre_region/ct/volume-7.nii'
    path1 = 'dataset/pre_region/seg/segmentation-7.nii'

    ct_pre_cut = sitk.ReadImage(path2, sitk.sitkFloat32)
    ct_pre_region = sitk.ReadImage(path, sitk.sitkFloat32)
    seg_pre_cut = sitk.ReadImage(path3, sitk.sitkUInt8)
    seg_pre_region = sitk.ReadImage(path1, sitk.sitkUInt8)


    pre_ct_cut_array = sitk.GetArrayFromImage(ct_pre_cut)
    pre_ct_region_array = sitk.GetArrayFromImage(ct_pre_region)
    pre_seg_cut_array = sitk.GetArrayFromImage(seg_pre_cut)
    pre_seg_region_array = sitk.GetArrayFromImage(seg_pre_region)

    pre_ct_cut_array = pre_ct_cut_array[None, ...]
    pre_ct_region_array = pre_ct_region_array[None, ...]
    pre_seg_cut_array = pre_seg_cut_array[None, ...]
    pre_seg_region_array = pre_seg_region_array[None, ...]

    pre_ct_cut_array, pre_seg_cut_array = CenterCrop([128,64,64])([pre_ct_cut_array,pre_seg_cut_array])
    pre_ct_region_array, pre_seg_region_array = CenterCrop([128,64,64])([pre_ct_region_array,pre_seg_region_array])
    # print(pre_ct_cut_array.shape, pre_seg_cut_array.shape)
    pre_array = np.vstack((pre_ct_region_array,pre_ct_cut_array,pre_seg_cut_array))
    print(pre_array.shape)

    plt.subplot(161)
    plt.imshow(pre_ct_region_array[0][53],cmap='gray')
    plt.subplot(162)
    plt.imshow(pre_ct_cut_array[0][53],cmap='gray')
    plt.subplot(163)
    plt.imshow(pre_seg_cut_array[0][53],cmap='gray')
    plt.subplot(164)
    plt.imshow(pre_array[0][53],cmap='gray')
    plt.subplot(165)
    plt.imshow(pre_array[1][53],cmap='gray')
    plt.subplot(166)
    plt.imshow(pre_array[2][53],cmap='gray')
    plt.show()

    # net = DiT_basic(basic_model='t2t',  # 使用vit还是t2t vit只支持both 因为其他的不需要消融探究
    #                 patch_emb='isolated',
    #                 time_emb=True,
    #                 pos_emb='share',
    #                 use_scale=True,
    #                 loss_f='focal',  # ce  # focal of ce
    #                 input_type='both',
    #                 pool='cls',
    #                 output_type='probability')
    # net = DiT_basic(basic_model='t2t')
    # output = net(pre_ct_array, pre_ct_array)
    # print(output)


if __name__ == '__main__':
    main()

