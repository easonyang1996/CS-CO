#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-16 17:33
# Email: yps18@mails.tsinghua.edu.cn
# Filename: 3_xie_train.py
# Description: 
#   the script to train the 2020miccai paper, Xie et al. 
# ******************************************************
import os
import sys
import time 
import numpy as np
import random
from tqdm import tqdm
import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim import lr_scheduler 
from torch.utils.data import DataLoader
from xie_dataset import get_training_set, get_validating_set 
from utils import EarlyStopping 

sys.path.append('..')
from model import Xie_miccai


class Triplet_Loss(nn.Module):
    def __init__(self, m1=1.0):
        super(Triplet_Loss, self).__init__()
        self.m1 = m1

    def _L2distance(self, x1, x2):
        return torch.sum((x1-x2)**2, dim=1)

    def forward(self, z_a, z_p, z_n):
        d_ap = self._L2distance(z_a, z_p)
        d_an = self._L2distance(z_a, z_n)
        loss = torch.mean(F.relu(d_ap-d_an+self.m1))
        return loss

class Count_Loss(nn.Module):
    def __init__(self, m2=1.0):
        super(Count_Loss, self).__init__()
        self.m2 = m2

    def forward(self, f_p, f_n):
        return torch.mean(torch.relu(f_n-f_p+self.m2))

class Xie_Loss(nn.Module):
    def __init__(self, m1=1.0, m2=1.0):
        super(Xie_Loss, self).__init__()
        self.triplet_loss = Triplet_Loss(m1)
        self.count_loss = Count_Loss(m2)

    def forward(self, z_a, z_p, z_n, f_p, f_n):
        loss1 = self.triplet_loss(z_a, z_p, z_n)
        loss2 = self.count_loss(f_p, f_n)
        return loss1+loss2


def train_epoch(train_loader, model, loss_fn, optimizer, device, epoch):
    model.train()
    losses = []
    p_bar = tqdm(train_loader)
    for anch, pos, neg in p_bar:
        anch = anch.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
       
        optimizer.zero_grad()
        z_a, z_p, z_n, f_p, f_n = model(anch, pos, neg)
        loss_outputs = loss_fn(z_a, z_p, z_n, f_p, f_n)

        losses.append(loss_outputs.item())
            
        loss_outputs.backward()
        optimizer.step()

        p_bar.set_description('Epoch {}'.format(epoch))
        p_bar.set_postfix(loss=loss_outputs.item())
                
    print('Epoch: {}\ttotal_loss {:.6f}'.format(epoch, np.mean(losses)))


def eval_epoch(eval_loader, model, loss_fn, device, epoch, early_stopping=None):
    with torch.no_grad():
        model.eval()
        val_loss = []
        p_bar = tqdm(eval_loader)
        for anch, pos, neg in p_bar:
            anch = anch.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
       
            optimizer.zero_grad()
            z_a, z_p, z_n, f_p, f_n = model(anch, pos, neg)
            loss_outputs = loss_fn(z_a, z_p, z_n, f_p, f_n)

            val_loss.append(loss_outputs.item())

            p_bar.set_description('Epoch {}'.format(epoch))
            p_bar.set_postfix(loss=loss_outputs.item())

    print('val Loss {:.6f}'.format(np.mean(val_loss)))    
    early_stopping(np.mean(val_loss), model, epoch)
    return np.mean(val_loss)


def data_list(path, k=None):
    img_list = os.listdir(path+'patches/')
    random.seed(10)
    random.shuffle(img_list)
    if k==None:
        return img_list
    else:
        return img_list[:k]


if __name__ == '__main__':
    # define gpu devices
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # parse config file
    assert len(sys.argv)==2, 'Please claim the config file!'
    print('config file in {}'.format(sys.argv[1]))
    cf = configparser.ConfigParser()
    cf.read(sys.argv[1])
    method_name = cf.sections()[0]
    print('training the {} model!'.format(method_name))

    train_root = cf.get(method_name, 'train_root')
    training_size = eval(cf.get(method_name, 'training_size'))
    valid_root = cf.get(method_name, 'valid_root')
    checkpoint_path = cf.get(method_name, 'checkpoint_path')
    if os.path.exists(checkpoint_path) == False:
        os.makedirs(checkpoint_path)
    backbone_name = cf.get(method_name, 'backbone_name')
    backbone_pretrain = cf.getboolean(method_name, 'backbone_pretrain')
    in_channel = cf.getint(method_name, 'in_channel')
    epochs = cf.getint(method_name, 'epochs')
    batch_size = cf.getint(method_name, 'batch_size')
    optim_name = cf.get(method_name, 'optim')
    lr = cf.getfloat(method_name, 'lr')
    weight_decay = cf.getfloat(method_name, 'weight_decay')
    

    training_list = data_list(train_root, k=training_size)
    model = Xie_miccai(network=backbone_name, in_channel=in_channel, pretrained=backbone_pretrain)

    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        if optim_name[0] == 'S':
            optimizer = optim.SGD(model.module.parameters(),
                                  lr=lr, momentum=0.9,
                                  weight_decay=weight_decay)
        if optim_name[0] == 'A':
            optimizer = optim.Adam(model.module.parameters(),
                                   lr=lr, weight_decay=weight_decay)
            
    else:
        if optim_name[0] == 'S':
            optimizer = optim.SGD(model.parameters(),
                                  lr=lr, momentum=0.9,
                                  weight_decay=weight_decay)
        if optim_name[0] == 'A':
            optimizer = optim.Adam(model.parameters(),
                                   lr=lr, weight_decay=weight_decay)
                    
    model.to(device)
   
    if optim_name.split('-')[-1] == 'step':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    if optim_name.split('-')[-1] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, verbose=True)
    
    ''' 
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1,
                                              total_epoch=5,
                                              after_scheduler=scheduler)
    '''
   

    loss_fn = Xie_Loss()
    train_dataset = get_training_set(train_root, training_list)
    eval_dataset = get_validating_set(valid_root) 

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False, num_workers=4)

    early_stopping = EarlyStopping(patience=15, verbose=True,
                                   path=checkpoint_path+'{}_{}_{}_{}_{}_{}.pth'.format(method_name,optim_name,training_size,batch_size,lr,weight_decay))
    
    
    for epoch in range(epochs):
        print('epoch {}/{}'.format(epoch+1, epochs))
        train_epoch(train_loader, model, loss_fn, optimizer, device, epoch+1)
        eval_loss = eval_epoch(eval_loader, model, loss_fn, device, epoch+1, early_stopping)
            
        if optim_name.split('-')[-1] == 'step':
            scheduler.step(eval_loss)  
        if optim_name.split('-')[-1] == 'cosine':
            scheduler.step()  
        ''' 
        if early_stopping.early_stop:
            print('Early stop!')
            break
        '''
    


