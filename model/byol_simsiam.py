#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-12 11:10
# Email: yps18@mails.tsinghua.edu.cn
# Filename: byol_simsiam.py
# Description: 
#       byol and simsiam implementation 
# ******************************************************
import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T
from torchvision.models import resnet18, resnet50

BACKBONE = {'resnet18': resnet18, 'resnet50': resnet50}

# helper functions


def flatten(t):
    return t.reshape(t.shape[0], -1)

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

def get_module_device(module):
    return next(module.parameters()).device

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

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2,
                 return_embedding=False):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.return_embedding = return_embedding

        self.hidden = None
        self.hook_registered = False 

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)

        if self.return_embedding:
            return representation
        
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection

# main class

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99,
        use_momentum = True,
        return_embedding = False
    ):
        super().__init__()
        self.return_embedding = return_embedding
        self.online_encoder = NetWrapper(net, projection_size,
                                         projection_hidden_size,
                                         layer=hidden_layer,
                                         return_embedding=return_embedding)
        
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device),
                     torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, image_one, image_two=None, return_embedding=False):
        if self.return_embedding:
            return self.online_encoder(image_one)

        online_proj_one = self.online_encoder(image_one)
        online_proj_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one = target_encoder(image_one)
            target_proj_two = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        return online_pred_one, online_pred_two, target_proj_one, target_proj_two


def Byol(network, img_size, moving_average_decay=0.99, use_momentum=True,
         backbone_pretrain=False, return_embedding=False):
    assert use_momentum, 'byol must use momentum!'
    backbone = BACKBONE[network](pretrained=backbone_pretrain)
    return BYOL(net=backbone, image_size=img_size,
                moving_average_decay=moving_average_decay,
                use_momentum=use_momentum, return_embedding=return_embedding)

def Simsiam(network, img_size, moving_average_decay=0.99, use_momentum=False,
            backbone_pretrain=False, return_embedding=False):
    assert not use_momentum, 'simsiam does not need momentum!'
    backbone = BACKBONE[network](pretrained=backbone_pretrain)
    return BYOL(net=backbone, image_size=img_size,
                moving_average_decay=moving_average_decay,
                use_momentum=use_momentum, return_embedding=return_embedding)
