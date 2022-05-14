#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-12 13:10
# Email: yps18@mails.tsinghua.edu.cn
# Filename: byol_train.py
# Description: 
#   the script to train byol and simsiam 
# ******************************************************
import os
import sys
import time 
import numpy as np
import random
import parser
from tqdm import tqdm
import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp 

from torch.optim import lr_scheduler 
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from byol_dataset import get_training_set, get_validating_set 
from utils import EarlyStopping 

sys.path.append('..')
from model import Byol, Simsiam
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'

METHODS = {'byol': Byol, 'simsiam': Simsiam}

def cleanup():
    dist.destroy_process_group()

class BYOL_Loss(nn.Module):
    def __init__(self):
        super(BYOL_Loss, self).__init__()

    def forward(self, opred1, opred2, tproj1, tproj2):
        opred1 = F.normalize(opred1, dim=-1, p=2)
        opred2 = F.normalize(opred2, dim=-1, p=2)
        tproj1 = F.normalize(tproj1.detach(), dim=-1, p=2)
        tproj2 = F.normalize(tproj2.detach(), dim=-1, p=2)
        loss_part1 = 2 - 2*(opred1*tproj2).sum(dim=-1)
        loss_part2 = 2 - 2*(opred2*tproj1).sum(dim=-1)
        loss = 0.5*loss_part1 + 0.5*loss_part2
        
        return loss.mean()

def train_epoch(train_loader, model, loss_fn, optimizer, gpu, epoch):
    model.train()
    losses = []
    p_bar = tqdm(train_loader)
    for patches, aug_patches in p_bar:
        patches = patches.to(gpu, non_blocking=False)
        aug_patches = aug_patches.to(gpu, non_blocking=False)
       
        optimizer.zero_grad()
        opred1, opred2, tproj1, tproj2 = model(patches, aug_patches)
        loss_outputs = loss_fn(opred1, opred2, tproj1, tproj2)

        losses.append(loss_outputs.item())
            
        loss_outputs.backward()
        optimizer.step()

        p_bar.set_description('Epoch {}'.format(epoch))
        p_bar.set_postfix(loss=loss_outputs.item())
        if model.module.use_momentum!=False:
            model.module.update_moving_average() 
        
    print('Epoch: {}\ttotal_loss {:.6f}'.format(epoch, np.mean(losses)))


def eval_epoch(eval_loader, model, loss_fn, gpu, epoch, early_stopping=None):
    with torch.no_grad():
        model.eval()
        val_loss = []
        p_bar = tqdm(eval_loader)
        for patches, aug_patches in p_bar:
            patches = patches.to(gpu, non_blocking=False)
            aug_patches = aug_patches.to(gpu, non_blocking=False)
            
            opred1, opred2, tproj1, tproj2 = model(patches, aug_patches)
            loss_outputs = loss_fn(opred1, opred2, tproj1, tproj2)

            val_loss.append(loss_outputs.item())

            p_bar.set_description('Epoch {}'.format(epoch))
            p_bar.set_postfix(loss=loss_outputs.item())

    print('val Loss {:.6f}'.format(np.mean(val_loss)))    
    #if early_stopping != None:
    #    early_stopping(np.mean(val_loss), model, epoch, ddp=True)
    return np.mean(val_loss)


def data_list(path, k=None):
    img_list = os.listdir(path+'patches/')
    random.seed(10)
    random.shuffle(img_list)
    if k==None:
        return img_list
    else:
        return img_list[:k]


def main(gpu, args):
    rank = args['nr'] * args['gpus'] + gpu
    dist.init_process_group('nccl', rank=rank, world_size=args['world_size'])
    torch.cuda.set_device(gpu)

    method_name = args['method_name']
    train_root = args['train_root']
    training_size = args['training_size']
    valid_root = args['valid_root']
    checkpoint_path = args['checkpoint_path']
    backbone_name = args['backbone_name']
    backbone_pretrain = args['backbone_pretrain']
    img_size = args['img_size']
    epochs = args['epochs']
    batch_size = args['batch_size']
    optim_name = args['optim_name']
    lr = args['lr']
    weight_decay = args['weight_decay']
    moving_average_decay = args['moving_average_decay']
    use_momentum = args['use_momentum']
    training_list = args['training_list']

    model = METHODS[method_name](network=backbone_name, img_size=img_size,
                                 moving_average_decay=moving_average_decay,
                                 use_momentum=use_momentum,
                                 backbone_pretrain=backbone_pretrain) 
    
    model = model.to(gpu)
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    '''
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
    '''
    if optim_name[0] == 'S':
        optimizer = optim.SGD(model.parameters(),
                              lr=lr, momentum=0.9, weight_decay=weight_decay)
    if optim_name[0] == 'A':
        optimizer = optim.Adam(model.parameters(),
                               lr=lr, weight_decay=weight_decay)
                    
   
    if optim_name.split('-')[-1] == 'step':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    if optim_name.split('-')[-1] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, verbose=True)
    
    ''' 
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1,
                                              total_epoch=5,
                                              after_scheduler=scheduler)
    '''
   

    loss_fn = BYOL_Loss()
    train_dataset = get_training_set(train_root, training_list)
    eval_dataset = get_validating_set(valid_root) 

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args['world_size'],
                                                                    rank=rank)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,
                                                                   num_replicas=args['world_size'],
                                                                   rank=rank,
                                                                   shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, num_workers=4,
                              pin_memory=True, sampler=train_sampler)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                             drop_last=False, num_workers=4,
                             pin_memory=True, sampler=eval_sampler)
    if gpu == 0:
        early_stopping = EarlyStopping(patience=15, verbose=True,
                                       path=checkpoint_path+'{}_{}_{}_{}_{}_{}_{}.pth'.format(method_name,optim_name,training_size,batch_size,lr,weight_decay,moving_average_decay))
    else:
        early_stopping = None
        
    for epoch in range(epochs):
        print('epoch {}/{}'.format(epoch+1, epochs))
            
        train_sampler.set_epoch(epoch)
        train_epoch(train_loader, model, loss_fn, optimizer, gpu, epoch+1)
        eval_loss = eval_epoch(eval_loader, model, loss_fn, gpu, epoch+1, early_stopping)
        mean_eval_loss = torch.tensor(eval_loss/args['gpus']).to(gpu)
        dist.barrier()
        dist.all_reduce(mean_eval_loss)
        print('gpu {} eval_loss:{}, mean_loss:{}'.format(gpu, eval_loss,
                                                         mean_eval_loss.cpu().numpy()))

        if optim_name.split('-')[-1] == 'step':
            scheduler.step(eval_loss)  
        elif optim_name.split('-')[-1] == 'cosine':
            scheduler.step() 

        if gpu == 0:
            early_stopping(mean_eval_loss.cpu().numpy(), model, epoch+1)
        '''
        if gpu == 0 and early_stopping.early_stop:
            print('Early stop!')
            break
        '''
    if gpu == 0:
        torch.save(model.module.state_dict(),
                   checkpoint_path+'{}_{}_{}_{}_{}_{}_{}_{}_{:.5f}.pth'.format(method_name,optim_name,training_size,batch_size,lr,weight_decay,moving_average_decay,epoch+1,mean_eval_loss.cpu().numpy()))
    cleanup()
    




if __name__ == '__main__':
    # define gpu devices
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # parse config file
    assert len(sys.argv)==3, 'Please claim the config file and num of GPU!'
    print('config file in {}'.format(sys.argv[1]))

    args = {}
    args['gpus'] = int(sys.argv[2])
    args['nr'] = 0
    args['world_size'] = args['gpus'] 
    cf = configparser.ConfigParser()
    cf.read(sys.argv[1])
    method_name = cf.sections()[0]
    print('training the {} model!'.format(method_name))

    args['method_name'] = method_name
    args['train_root'] = cf.get(method_name, 'train_root')
    args['training_size'] = eval(cf.get(method_name, 'training_size'))
    args['valid_root'] = cf.get(method_name, 'valid_root')
    args['checkpoint_path'] = cf.get(method_name, 'checkpoint_path')
    if os.path.exists(args['checkpoint_path']) == False:
        os.makedirs(args['checkpoint_path'])
    args['backbone_name'] = cf.get(method_name, 'backbone_name')
    args['backbone_pretrain'] = cf.getboolean(method_name, 'backbone_pretrain')
    args['img_size'] = cf.getint(method_name, 'img_size')
    args['epochs'] = cf.getint(method_name, 'epochs')
    args['batch_size'] = cf.getint(method_name, 'batch_size')
    args['optim_name'] = cf.get(method_name, 'optim')
    args['lr'] = cf.getfloat(method_name, 'lr')
    args['weight_decay'] = cf.getfloat(method_name, 'weight_decay')
    args['moving_average_decay'] = cf.getfloat(method_name, 'moving_average_decay')
    args['use_momentum'] = cf.getboolean(method_name, 'use_momentum')

    args['training_list'] = data_list(args['train_root'],
                                      k=args['training_size'])
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    
    mp.spawn(main, args=(args, ), nprocs=args['gpus'])

    

