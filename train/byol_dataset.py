#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-14 15:48
# Email: yps18@mails.tsinghua.edu.cn
# Filename: 1_byol_dataset.py
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


class DatasetFromFolder(Dataset):
    def __init__(self, data_dir, img_list):
        super(DatasetFromFolder, self).__init__()
        self.data_dir = data_dir 
        self.img_list = img_list

    def __getitem__(self, index):

        img_name = self.img_list[index]
        patch = Image.open(os.path.join(self.data_dir+'patches/', img_name))
        
        DEFAULT_AUG = transforms.Compose([
            T.RandomResizedCrop((224, 224), (0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply(nn.ModuleList([T.ColorJitter(0.4, 0.4, 0.4, 0.1)]),
                          p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur((3, 3), (0.1, 2.0)),
            T.ToTensor()
            #T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
            #            std=torch.tensor([0.229, 0.224, 0.225])),
        ])
        
        patch1 = DEFAULT_AUG(patch)
        patch2 = DEFAULT_AUG(patch)

        return patch1, patch2

    def __len__(self):
        return len(self.img_list)

