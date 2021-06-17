#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-15 14:20
# Email: yps18@mails.tsinghua.edu.cn
# Filename: 2_chen_dataset.py
# Description: 
#   custom dataset from folder
# ******************************************************
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from torchvision import transforms as T

import numpy as np
from PIL import Image
import os
import random

'''
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)
'''

def get_training_set(train_dir, train_list=None):
    if train_list == None:
        patches_dir = train_dir + 'patches/'
        assert os.path.exists(patches_dir)
        img_list = os.listdir(patches_dir)
    else:
        img_list = train_list
    return DatasetFromFolder(train_dir, img_list)

def get_validating_set(valid_dir, valid_list=None):
    if valid_list == None:
        patches_dir = valid_dir + 'patches/'
        assert os.path.exists(patches_dir)
        img_list = os.listdir(patches_dir)
        img_list.sort()
    else:
        img_list = valid_list
    return DatasetFromFolder(valid_dir, img_list)

def context_disordering(img, iteration=10):
    img = np.array(img)
    height, width = img.shape[0], img.shape[1]
    crop_size = int(height*0.1)

    for i in range(iteration):
        h1 = random.randint(0, height-crop_size)
        w1 = random.randint(0, width-crop_size)
        flag_h = 0       # if h2 is good?
        flag_w = 0       # if w2 is good?

        while 1:
            h2 = random.randint(0, height-crop_size)
            w2 = random.randint(0, width-crop_size)

            if h2>h1+crop_size or h2<h1-crop_size:
                flag_h = 1
            if w2>w1+crop_size or w2<w1-crop_size:
                flag_w = 1
            
            if flag_h==1 and flag_w==1:
                context1 = img[h1:h1+crop_size, w1:w1+crop_size].copy() #copy is necessary
                context2 = img[h2:h2+crop_size, w2:w2+crop_size].copy()
                img[h1:h1+crop_size, w1:w1+crop_size] = context2
                img[h2:h2+crop_size, w2:w2+crop_size] = context1
                break
            else:
                continue

    return Image.fromarray(img)


class DatasetFromFolder(Dataset):
    def __init__(self, data_dir, img_list):
        super(DatasetFromFolder, self).__init__()
        self.data_dir = data_dir 
        self.img_list = img_list

    def __getitem__(self, index):

        img_name = self.img_list[index]
        patch = Image.open(os.path.join(self.data_dir+'patches/', img_name))
       
        disorder_patch = context_disordering(patch)
        t = T.ToTensor()
        return t(patch), t(disorder_patch)

    def __len__(self):
        return len(self.img_list)

