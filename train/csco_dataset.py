#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-18 14:00
# Email: yps18@mails.tsinghua.edu.cn
# Filename: csco_dataset.py
# Description: 
#   custom dataset from folder
# ******************************************************
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image
import os
import random
import cv2
import time



def my_transforms(H_prime, E_prime):

    if random.random()>0.5:
        H_prime = TF.vflip(H_prime)
        E_prime = TF.vflip(E_prime)
    if random.random()>0.5:
        H_prime = TF.hflip(H_prime)
        E_prime = TF.hflip(E_prime)
    
    if random.random()>0.5:
        crop_ratio = random.uniform(0.7,0.9)
        size = H_prime.size[0]
        crop_size = int(size*crop_ratio)
        H_prime = TF.resize(TF.center_crop(H_prime, crop_size), size)
        E_prime = TF.resize(TF.center_crop(E_prime, crop_size), size)
    
    if random.random()>0.7:
        H_prime = TF.gaussian_blur(H_prime, (3,3), (1.0,2.0))
        E_prime = TF.gaussian_blur(E_prime, (3,3), (1.0,2.0))
    
    return TF.to_tensor(H_prime), TF.to_tensor(E_prime)


def get_training_set(train_dir, train_list=None, model_type='cs'):
    if train_list == None:
        patches_dir = train_dir + 'patches/'
        assert os.path.exists(patches_dir)
        img_list = os.listdir(patches_dir)
    else:
        img_list = train_list 
    return DatasetFromFolder(train_dir, img_list, model_type)

def get_validating_set(valid_dir, valid_list=None, model_type='cs'):
    if valid_list == None:
        patches_dir = valid_dir + 'patches/'
        assert os.path.exists(patches_dir)
        img_list = os.listdir(patches_dir)
    else:
        img_list = valid_list
    return DatasetFromFolder(valid_dir, img_list, model_type)


class DatasetFromFolder(Dataset):
    def __init__(self, data_dir, img_list, model_type):
        super(DatasetFromFolder, self).__init__()
        self.data_dir = data_dir 
        self.img_list = img_list
        self.model_type = model_type 

    def __getitem__(self, index):
        img_name = self.img_list[index].split('.')[0]
        H_ori = Image.open(os.path.join(self.data_dir+'H/', img_name+'_H.png'))
        E_ori = Image.open(os.path.join(self.data_dir+'E/', img_name+'_E.png'))
        if self.model_type == 'cs':
            tt = transforms.ToTensor()
            return tt(H_ori), tt(E_ori), img_name
        H_prime_ori = Image.open(os.path.join(self.data_dir+'H_prime/',
                                          img_name+'_H_prime.png'))
        E_prime_ori = Image.open(os.path.join(self.data_dir+'E_prime/',
                                          img_name+'_E_prime.png'))
        
        H, E = my_transforms(H_ori, E_ori)
        H_prime, E_prime = my_transforms(H_prime_ori, E_prime_ori)

        return H, E, H_prime, E_prime, img_name 

    def __len__(self):
        return len(self.img_list)

