#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-16 17:29
# Email: yps18@mails.tsinghua.edu.cn
# Filename: xie_dataset.py
# Description: 
#   custom dataset for xie_miccai
# ******************************************************
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import numpy as np
from PIL import Image
import os
import random


def get_training_set(train_dir, train_list=None):
    if train_list is None:
        patches_dir = train_dir + 'patches/'
        assert os.path.exists(patches_dir)
        img_list = os.listdir(patches_dir)
    else:
        img_list = train_list 
    return DatasetFromFolder(train_dir, img_list)

def get_validating_set(valid_dir, valid_list=None):
    if valid_list is None:
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
        patch = Image.open(os.path.join(self.data_dir+'patches/', self.img_list[index]))
        
        pos_transform = transforms.RandomCrop(170)
        neg_transform = transforms.RandomChoice([transforms.RandomCrop(112),
                                                 transforms.RandomCrop(56)])
        totensor_transform = transforms.Compose([transforms.Resize(224), 
                                                 transforms.ToTensor()])
        
        anch = pos_transform(patch)
        pos = pos_transform(patch)
        neg = neg_transform(pos)
    
        anch = totensor_transform(anch)
        pos =  totensor_transform(pos)
        neg = totensor_transform(neg)
        return anch, pos, neg 

    def __len__(self):
        return len(self.img_list)

