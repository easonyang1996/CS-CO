#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-14 14:32
# Email: yps18@mails.tsinghua.edu.cn
# Filename: chen_MIA.py
# Description: 
#   reproduce of the Medical Image analysis paper:
#       self-supervised learning for medical image analysis
#       using image context restoration.
# ******************************************************

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18, resnet50
import numpy as np

BACKBONE = {'resnet18': resnet18, 'resnet50': resnet50}

class Encoder(nn.Module):
    def __init__(self, encoder_name, pretrained=False, in_channel=3):
        super(Encoder, self).__init__()
        self.backbone = BACKBONE[encoder_name](half_channel=False,
                                               in_channel=in_channel)

    def forward(self, x):
        x = self.backbone(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoder_name, out_channel):
        super(Decoder, self).__init__()
        if encoder_name[-2:] in ['18', '34']:
            decoder_channel = [512, 256, 128, 64, 64]
        else:
            decoder_channel = [2048, 1024, 512, 256, 64]

        self.up1 = nn.ConvTranspose2d(decoder_channel[0], decoder_channel[1],
                                      2, stride=2)
        self.up2 = nn.ConvTranspose2d(decoder_channel[1], decoder_channel[2],
                                      2, stride=2)
        self.up3 = nn.ConvTranspose2d(decoder_channel[2], decoder_channel[3],
                                      2, stride=2)
        self.up4 = nn.ConvTranspose2d(decoder_channel[3], decoder_channel[4],
                                      2, stride=2)
        self.up5 = nn.ConvTranspose2d(decoder_channel[4], out_channel, 2, 
                                      stride=2)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x


class CHEN_MIA(nn.Module):
    def __init__(self, encoder_name, in_channel, pretrained=False,
                 return_embedding=False):
        super(CHEN_MIA, self).__init__()
        self.return_embedding = return_embedding 
        
        self.encoder = Encoder(encoder_name, pretrained, in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if not return_embedding:
            self.decoder = Decoder(encoder_name, in_channel)

    def forward(self, x):
        enco_x = self.encoder(x)

        if self.return_embedding:
            embedding = self.avgpool(enco_x)
            embedding = torch.flatten(embedding, 1)
            return embedding

        deco_x = self.decoder(enco_x)

        return deco_x


def Chen_mia(network, in_channel, pretrained=False, return_embedding=False):
    return CHEN_MIA(encoder_name=network, in_channel=in_channel,
                    pretrained=pretrained, return_embedding=return_embedding)

