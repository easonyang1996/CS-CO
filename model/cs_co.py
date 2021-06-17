#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-13 17:58
# Email: yps18@mails.tsinghua.edu.cn
# Filename: cs_co.py
# Description: 
#       cross stain contrastive learning 
# ******************************************************

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .model_parts import Up, OutConv
import numpy as np
import copy
from functools import wraps

BACKBONE = {'resnet18': resnet18, 'resnet34': resnet34, 
            'resnet50': resnet50, 'resnet101': resnet101, 
            'resnet152': resnet152}

# helper functions


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

'''
def get_module_device(module):
    return next(module.parameters()).device
'''

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


###################################### sub module ###################################

class Encoder(nn.Module):
    def __init__(self, encoder_name, pretrained=False, half_channel=False,
                 in_channel=3):
        super(Encoder, self).__init__()
        self.backbone = BACKBONE[encoder_name](pretrained=pretrained,
                                                half_channel=half_channel,
                                                in_channel=in_channel)

    def forward(self, x):
        x = self.backbone(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoder_name, out_channel, half_channel=False,
                 bilinear=True, decoder_freeze=False):
        super(Decoder, self).__init__()
        if encoder_name[-2:] in ['18', '34']:   
            decoder_channel = np.array([512, 256, 128, 64, 64])
        else:
            decoder_channel = np.array([2048, 1024, 512, 256, 64]) 

        if half_channel==True:
            decoder_channel = decoder_channel // 2

        self.up1 = Up(decoder_channel[0], decoder_channel[1], bilinear=bilinear)
        self.up2 = Up(decoder_channel[1], decoder_channel[2], bilinear=bilinear)
        self.up3 = Up(decoder_channel[2], decoder_channel[3], bilinear=bilinear)
        self.up4 = Up(decoder_channel[3], decoder_channel[4], bilinear=bilinear)
        self.up5 = Up(decoder_channel[4], out_channel, bilinear=bilinear)

        if decoder_freeze:
            for p in self.parameters():
                p.requires_grad = False


    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x


class HE_Encoder(nn.Module):
    def __init__(self, encoder_name, in_channel, pretrained=False,
                 half_channel=False):
        super(HE_Encoder, self).__init__()
        self.H2E_encoder = Encoder(encoder_name, pretrained=pretrained,
                                   half_channel=half_channel,
                                   in_channel=in_channel)
        self.E2H_encoder = Encoder(encoder_name, pretrained=pretrained,
                                   half_channel=half_channel,
                                   in_channel=in_channel)

    def forward(self, h, e):
        h_out = self.H2E_encoder(h)
        e_out = self.E2H_encoder(e)
        return h_out, e_out


class HE_Decoder(nn.Module):
    def __init__(self, encoder_name, in_channel, half_channel=False,
                 bilinear=True, decoder_freeze=False):
        super(HE_Decoder, self).__init__()
        self.H2E_decoder = Decoder(encoder_name, out_channel=in_channel,
                                   half_channel=half_channel,
                                   bilinear=bilinear, decoder_freeze=decoder_freeze)
        self.E2H_decoder = Decoder(encoder_name, out_channel=in_channel,
                                   half_channel=half_channel,
                                   bilinear=bilinear, decoder_freeze=decoder_freeze)

    def forward(self, h_out, e_out):
        e_pred = self.H2E_decoder(h_out)
        h_pred = self.E2H_decoder(e_out)
        return e_pred, h_pred


# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


####################################### main module ##################################
# main class
class CS_CO(nn.Module):
    def __init__(self, encoder_name, in_channel, projection_size=256,
                 model_type='cs', bilinear=True, half_channel=False,
                 decoder_freeze=False, pretrained=False, 
                 pretrained_recon=None, moving_average_decay=0.99, 
                 use_momentum=True, return_embedding=False):
        super(CS_CO, self).__init__()

        self.model_type = model_type
        self.use_momentum = use_momentum
        self.return_embedding = return_embedding
        
        self.online_he_encoder = HE_Encoder(encoder_name, in_channel, pretrained,
                                         half_channel) 
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        if not return_embedding:
            self.he_decoder = HE_Decoder(encoder_name, in_channel, half_channel,
                                         bilinear, decoder_freeze) 

            if pretrained_recon != 'None':
                self._load_encoder_weight(pretrained_recon)
                self._load_decoder_weight(pretrained_recon)

            if model_type == 'cs-co':
                mlp_dim = 1024 if encoder_name[-2:] in ['18', '34'] else 2048
                mlp_dim = mlp_dim//2 if half_channel else mlp_dim
                
                self.online_projector = MLP(dim=mlp_dim,
                                            projection_size=projection_size,
                                            hidden_size=4096)
                self.online_predictor = MLP(dim=projection_size,
                                            projection_size=projection_size,
                                            hidden_size=4096)

                self.target_encoder = None
                self.target_projector = None 
                self.target_ema_updater = EMA(moving_average_decay)


                init_h = torch.randn(2, in_channel, 224, 224)
                init_e = torch.randn(2, in_channel, 224, 224)
                init_h_prime = torch.randn(2, in_channel, 224, 224)
                init_e_prime = torch.randn(2, in_channel, 224, 224)

                self.forward(init_h, init_e, init_h_prime, init_e_prime)

    def _load_encoder_weight(self, weight_path):
        # load encoder weight
        pretrained_dict = torch.load(weight_path)
        pretrained_dict = {k[18:]:v for k,v in pretrained_dict.items() if
                           k[:18]=='online_he_encoder.'}
        self.online_he_encoder.load_state_dict(pretrained_dict)

    def _load_decoder_weight(self, weight_path):
        # load decoder weight
        pretrained_dict = torch.load(weight_path)
        pretrained_dict = {k[11:]:v for k,v in pretrained_dict.items() if
                           k[:11]=='he_decoder.'}
        self.he_decoder.load_state_dict(pretrained_dict)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_he_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder
    
    @singleton('target_projector')
    def _get_target_project(self):
        target_projector = copy.deepcopy(self.online_projector)
        set_requires_grad(target_projector, False)
        return target_projector

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None
        del self.target_projector
        self.target_projector = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder,
                              self.online_he_encoder)
        assert self.target_projector is not None, 'target projector has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_projector,
                              self.online_projector)

    def forward(self, h, e, h_prime=None, e_prime=None):
        online_enco_one_h, online_enco_one_e = self.online_he_encoder(h, e)
        
        if self.return_embedding:
            online_enco_one_h_pool = self.avgpool(online_enco_one_h)
            online_enco_one_e_pool = self.avgpool(online_enco_one_e)
            embedding = torch.cat([online_enco_one_h_pool,
                                   online_enco_one_e_pool], dim=1)
            embedding = torch.flatten(embedding, 1)
            return embedding

        pred_one_e, pred_one_h = self.he_decoder(online_enco_one_h,
                                                 online_enco_one_e)

        if self.model_type == 'cs':
            return pred_one_e, pred_one_h 

        if h_prime!=None:
            online_enco_two_h, online_enco_two_e =self.online_he_encoder(h_prime,
                                                                         e_prime)

        if self.model_type == 'cs-co':
            # h e 
            online_enco_one_h_pool = self.avgpool(online_enco_one_h)
            online_enco_one_e_pool = self.avgpool(online_enco_one_e)
            online_enco_one = torch.cat([online_enco_one_h_pool,
                                         online_enco_one_e_pool], dim=1)
            online_enco_one = torch.flatten(online_enco_one, 1)
            online_proj_one = self.online_projector(online_enco_one)
            online_pred_one = self.online_predictor(online_proj_one)
            
            # h_prime e_prime 
            online_enco_two_h_pool = self.avgpool(online_enco_two_h)
            online_enco_two_e_pool = self.avgpool(online_enco_two_e)
            online_enco_two = torch.cat([online_enco_two_h_pool,
                                         online_enco_two_e_pool], dim=1)
            online_enco_two = torch.flatten(online_enco_two, 1)
            online_proj_two = self.online_projector(online_enco_two)
            online_pred_two = self.online_predictor(online_proj_two)
            # target branch
            with torch.no_grad():
                target_encoder = self._get_target_encoder() if self.use_momentum else self.online_he_encoder
                target_projector = self._get_target_project() if self.use_momentum else self.online_projector
                 
                # h e 
                target_enco_one_h, target_enco_one_e = target_encoder(h, e)
                target_enco_one_h_pool = self.avgpool(target_enco_one_h)
                target_enco_one_e_pool = self.avgpool(target_enco_one_e)
                target_enco_one = torch.cat([target_enco_one_h_pool,
                                             target_enco_one_e_pool], dim=1)
                target_enco_one = torch.flatten(target_enco_one, 1)
                target_proj_one = target_projector(target_enco_one)
                target_proj_one.detach_()
            
                # h_prime e_prime 
                target_enco_two_h, target_enco_two_e = target_encoder(h_prime,
                                                                      e_prime)
                target_enco_two_h_pool = self.avgpool(target_enco_two_h)
                target_enco_two_e_pool = self.avgpool(target_enco_two_e)
                target_enco_two = torch.cat([target_enco_two_h_pool,
                                             target_enco_two_e_pool], dim=1)
                target_enco_two = torch.flatten(target_enco_two, 1)
                target_proj_two = target_projector(target_enco_two)
                target_proj_two.detach_()
            
            return online_pred_one, online_pred_two, target_proj_one, target_proj_two, pred_one_e, pred_one_h


def Cs_co(network, in_channel, model_type='cs', decoder_freeze=False, 
          pretrained_recon=None, moving_average_decay=0.99, 
          use_momentum=True, return_embedding=False):
    return CS_CO(encoder_name=network, in_channel=in_channel,
                 model_type=model_type, decoder_freeze=decoder_freeze,
                 pretrained_recon=pretrained_recon,
                 moving_average_decay=moving_average_decay,
                 use_momentum=use_momentum, return_embedding=return_embedding)

