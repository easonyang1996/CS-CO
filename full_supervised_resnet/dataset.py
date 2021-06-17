#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-14 13:24
# Email: yps18@mails.tsinghua.edu.cn
# Filename: dataset.py
# Description: 
#       dataset for full supervised resnet training and test 
# ******************************************************
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import numpy as np
from PIL import Image
import os


def get_training_set(train_dir, cls, train_list=None):
    if train_list is None:
        patches_dir = train_dir + 'patches/'
        assert os.path.exists(patches_dir)
        img_list = os.listdir(patches_dir)
    else:
        img_list = train_list
    return DatasetFromFolder(train_dir, img_list, cls)


def get_validating_set(valid_dir, cls, valid_list=None):
    if valid_list is None:
        patches_dir = valid_dir + 'patches/'
        assert os.path.exists(patches_dir)
        img_list = os.listdir(patches_dir)
        img_list.sort()
    else:
        img_list = valid_list
    return DatasetFromFolder(valid_dir, img_list, cls)


def get_testing_set(test_dir, cls, test_list=None):
    if test_list is None:
        patches_dir = test_dir + 'patches/'
        assert os.path.exists(patches_dir)
        img_list = os.listdir(patches_dir)
        img_list.sort()
    else:
        img_list = test_list
    return DatasetFromFolder(test_dir, img_list, cls)


class DatasetFromFolder(Dataset):
    def __init__(self, data_dir, img_list, cls):
        super(DatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.img_list = img_list
        self.cls = cls

    def __getitem__(self, index):
        label = self.img_list[index].split('-')[0]
        label = int(self.cls[label])        #get item's label

        patch = Image.open(os.path.join(self.data_dir+'patches/', self.img_list[index]))
        transform = transforms.ToTensor()
        patch = transform(patch)
        return patch, label

    def __len__(self):
        return len(self.img_list)






