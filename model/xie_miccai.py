#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-16 16:56
# Email: yps18@mails.tsinghua.edu.cn
# Filename: xie_miccai.py
# Description: 
#   reproduce of the miccai2020 paper:
#       Instance-aware Self-supervised Learning for Nuclei Segmentation
# ******************************************************
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import numpy as np 

BACKBONES = {'resnet18':resnet18, 'resnet34':resnet34,
             'resnet50':resnet50, 'resnet101':resnet101,
             'resnet152':resnet152}


class Encoder(nn.Module):
    def __init__(self, encoder_name, pretrained=False, in_channel=3):
        super(Encoder, self).__init__()
        self.backbone = BACKBONES[encoder_name](pretrained=pretrained,
                                                in_channel=in_channel)

    def forward(self, x):
        x = self.backbone(x)
        return x

class InsAwaSup(nn.Module):
    def __init__(self, encoder_name, in_channel, pretrained=False,
                 return_embedding=False):
        super(InsAwaSup, self).__init__()
        self.return_embedding = return_embedding 

        self.encoder = Encoder(encoder_name, pretrained, in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if encoder_name in ['resnet18', 'resnet34']:
            self.fc = nn.Linear(512, 128)
        else:
            self.fc = nn.Linear(2048, 128)
        if not return_embedding:
            self.count_scaler = nn.Linear(128, 1)

    def forward(self, anch, pos=None, neg=None):
        z_anch = self.encoder(anch)
        z_anch = self.avgpool(z_anch)
        z_anch = torch.flatten(z_anch, 1)
        z_anch = self.fc(z_anch)

        if self.return_embedding:
            return z_anch

        if pos is not None:
            assert self.return_embedding == False
            z_pos = self.encoder(pos)
            z_pos = self.avgpool(z_pos)
            z_pos = torch.flatten(z_pos, 1)
            z_pos = self.fc(z_pos)

            z_neg = self.encoder(neg)
            z_neg = self.avgpool(z_neg)
            z_neg = torch.flatten(z_neg, 1)
            z_neg = self.fc(z_neg)
            
            f_p = self.count_scaler(z_pos)
            f_n = self.count_scaler(z_neg)
        
            return z_anch, z_pos, z_neg, f_p, f_n

def Xie_miccai(network, in_channel, pretrained=False, return_embedding=False):
    return InsAwaSup(encoder_name=network, in_channel=in_channel,
                     pretrained=pretrained, return_embedding=return_embedding)
        
    
