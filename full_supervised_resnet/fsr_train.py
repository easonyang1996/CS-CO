#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-14 13:26
# Email: yps18@mails.tsinghua.edu.cn
# Filename: fsr_train.py
# Description: 
#   train the fully supervised resnet 
# ******************************************************
import os 
import numpy as np
import random
import argparse
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim 
import torch.nn as nn

import sklearn
from sklearn.model_selection import KFold 
import timm
from torchvision import models
from tqdm import tqdm

from utils import EarlyStopping
from dataset import get_training_set, get_validating_set, get_testing_set

import sys
sys.path.append('..')
from model import Linear, Simsiam, Byol, Cs_co 


#TRAIN_DATA = '../BioImage_2015_data/dataset/train/'
#TEST_DATA = '../BioImage_2015_data/dataset/init_test/'
TRAIN_DATA = '../../NCT_CRC_data/all_data/train/'
TEST_DATA = '../../NCT_CRC_data/all_data/test/'

def get_img_list(path, k=None):
    total_list = os.listdir(path)
    total_list.sort()
    random.seed(8)
    random.shuffle(total_list)
    if k is None:
        return total_list 
    else:
        return total_list[:k]

def train_data_info(img_list):
    info = defaultdict(int)
    for img in img_list:
        label = img.split('-')[0]
        info[label] += 1
    print(info)


def train_epoch(train_loader, model, optimizer, loss_fn, device, epoch):
    model.train()
    losses = []
    #total_loss = 0
    
    p_bar = tqdm(train_loader) 
    for batch_idx, (patches, y) in enumerate(p_bar):
        
        patches = patches.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_prob = model(patches)

        loss_outputs = loss_fn(y_prob, y)
        losses.append(loss_outputs.item())
        #total_loss += loss_outputs.item()
        
        if loss_outputs.requires_grad is True:
            loss_outputs.backward()
            optimizer.step()

        tmp_y_true = y.detach().cpu().numpy()
        tmp_y_pred = torch.argmax(y_prob, dim=1).detach().cpu().numpy()
        
        if batch_idx==0:
            y_true = tmp_y_true 
            y_pred = tmp_y_pred 
        else:
            y_true = np.concatenate([y_true, tmp_y_true])
            y_pred = np.concatenate([y_pred, tmp_y_pred])

        '''
        if batch_idx % 50 == 0:
            message = 'Train: [{}/{} ({:.2f}%)]\tLoss: {:.6f}, ACC: {:.6f}'.format(int(batch_idx*y.size(0)), len(train_loader.dataset), 100.*batch_idx/len(train_loader), np.mean(losses), np.mean(y_true==y_pred))
            print(message)
            losses = []
        '''
        p_bar.set_description('Epoch {}'.format(epoch))
        p_bar.set_postfix(loss=loss_outputs.item())
    print('Epoch: {}, total_loss: {:.6f}, acc: {:.6f}'.format(epoch, np.mean(losses), np.mean(y_true==y_pred)))


def eval_epoch(eval_loader, model, loss_fn, device, epoch, eval_type,
               early_stopping=None):
    with torch.no_grad():
        model.eval()
        val_loss = []
        p_bar = tqdm(eval_loader)
        for batch_idx, (patches, y) in enumerate(p_bar):
            patches = patches.to(device)
            y = y.to(device)
            
            y_prob = model(patches)

            loss_outputs = loss_fn(y_prob, y)
            val_loss.append(loss_outputs.item())

            tmp_y_true = y.detach().cpu().numpy()
            tmp_y_pred = torch.argmax(y_prob, dim=1).detach().cpu().numpy()
        
            if batch_idx==0:
                y_true = tmp_y_true 
                y_pred = tmp_y_pred 
            else:
                y_true = np.concatenate([y_true, tmp_y_true])
                y_pred = np.concatenate([y_pred, tmp_y_pred])
            
            p_bar.set_description('Epoch {}'.format(epoch))
            p_bar.set_postfix(loss=loss_outputs.item())

    print('{} loss: {:.6f}, acc: {:.6f}'.format(eval_type, np.mean(val_loss), np.mean(y_true==y_pred)))
    if eval_type == 'valid':
        early_stopping(np.mean(val_loss), model)
        return np.mean(val_loss)
    elif eval_type == 'test':
        return np.mean(y_true==y_pred)

            




if __name__ == '__main__': 
    # hyper-parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 500
    lr = 0.001
    train_data_size = None
    num_fold = 5

    # classes and index
    #cls = {'NOR': 0, 'BEN': 1, 'INS': 2, 'INV': 3}
    cls = {'ADI':0, 'DEB':1, 'LYM':2, 'MUC':3, 'MUS':4, 'NORM':5, 'STR':6, 'TUM':7}
    print(cls)
    
    total_train_list = np.array(get_img_list(TRAIN_DATA+'patches/',
                                             k=train_data_size))
    train_data_info(total_train_list)


    # linear classifier and cross-validation
    kf = KFold(n_splits=num_fold)
    results = []
    e_losses = []
    stop_epochs = []
    for i, (train_index, valid_index) in enumerate(kf.split(total_train_list)):
        model = models.resnet18(num_classes=len(cls)) 
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        #optimizer = optim.Adam(model.parameters(), lr=lr)

        train_dataset = get_training_set(TRAIN_DATA, cls,
                                         total_train_list[train_index])
        valid_dataset = get_validating_set(TRAIN_DATA, cls,
                                           total_train_list[valid_index])
        test_dataset = get_testing_set(TEST_DATA, cls)   

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                                  shuffle=False, drop_last=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, drop_last=False, num_workers=4)

        loss_fn = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
        ckpt_path = './checkpoint/resnet18_'+str(i)+'.pth'
        early_stopping = EarlyStopping(patience=15, verbose=True, delta=0.001,
                                       path=ckpt_path) 
    
        
        for epoch in range(1, epochs+1):
            train_epoch(train_loader, model, optimizer, loss_fn, device, epoch) 
            val_loss = eval_epoch(valid_loader, model, loss_fn, device, epoch, 'valid',
                                  early_stopping)
            scheduler.step(val_loss)
            if early_stopping.early_stop:
                print('Early Stop!')
                break
        
        model.load_state_dict(torch.load(ckpt_path))
        test_acc = eval_epoch(test_loader, model, loss_fn, device, 0,
                              'test')
        results.append(test_acc)
        e_losses.append(-early_stopping.best_score)
        stop_epochs.append(epoch)

        #break
    results = np.array(results)
    print('acc: ',results)
    print('loss: ',e_losses)
    print('stop_epochs: ', stop_epochs)
    print('5-fold test_acc, mean: {:.6f}, std: {:.6f}\n'.format(results.mean(),
                                                                results.std()))
    
    f = open('FullySupResult.txt', 'a')
    f.write('acc: {}\n'.format(results))
    f.write('test_acc, mean:{:.6f}, std: {:.6f}\n\n'.format(results.mean(),
                                                           results.std()))
    f.close()

