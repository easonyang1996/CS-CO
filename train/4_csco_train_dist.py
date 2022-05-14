#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-18 13:55
# Email: yps18@mails.tsinghua.edu.cn
# Filename: 4_csco_train.py
# Description: 
#   the script to train the cross stain contrastive learning medel
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
import torch.distributed as dist
import torch.multiprocessing as mp 

from torch.optim import lr_scheduler 
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP 
from csco_dataset import get_training_set, get_validating_set 
from utils import EarlyStopping 

sys.path.append('..')
from model import Cs_co

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'

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

class Recon_Loss(nn.Module):
    def __init__(self, l1orl2):
        super(Recon_Loss, self).__init__()
        if l1orl2 == 'l1':
            self.fn = nn.L1Loss()
        elif l1orl2 == 'l2':
            self.fn = nn.MSELoss()

    def forward(self, pred_e, pred_h, h, e):
        loss = self.fn(pred_e, e)+self.fn(pred_h, h)
        return loss

class Csco_Loss(nn.Module):
    def __init__(self, model_type='cs', l1orl2='l2', gamma1=1, gamma2=1):
        super(Csco_Loss, self).__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.model_type = model_type
        self.recon_loss = Recon_Loss(l1orl2)
        if model_type == 'cs-co':
            self.byol_loss = BYOL_Loss()

    def forward(self, **kwargs):
        pred_one_e = kwargs['pred_one_e']
        pred_one_h = kwargs['pred_one_h']
        h = kwargs['h']
        e = kwargs['e']
        part_recon = self.recon_loss(pred_one_e, pred_one_h, h, e)

        if self.model_type == 'cs-co':
            online_one = kwargs['online_one']
            online_two = kwargs['online_two']
            target_one = kwargs['target_one']
            target_two = kwargs['target_two']
            part_byol = self.byol_loss(online_one, online_two, target_one, target_two)
            return self.gamma1*part_byol+self.gamma2*part_recon, part_byol.item(), part_recon.item()

        return part_recon, part_recon.item()


def train_epoch(train_loader, model, model_type, loss_fn, optimizer, gpu, epoch):
    model.train()
    losses = []
    p_bar = tqdm(train_loader)
    if model_type == 'cs':
        for H, E, _ in p_bar:
            H = H.to(gpu, non_blocking=False)
            E = E.to(gpu, non_blocking=False)
            
            optimizer.zero_grad()
            pred_one_e, pred_one_h = model(H, E)
            loss_outputs, _ = loss_fn(pred_one_e=pred_one_e, pred_one_h=pred_one_h,
                                   h=H, e=E)

            losses.append(loss_outputs.item())
            
            loss_outputs.backward()
            optimizer.step()

            p_bar.set_description('Epoch {}'.format(epoch))
            p_bar.set_postfix(loss=loss_outputs.item())
            if model.module.use_momentum!=False:
                model.module.update_moving_average() 
    
    elif model_type == 'cs-co':
        for H, E, H_prime, E_prime, _ in p_bar:
            H = H.to(gpu, non_blocking=False)
            E = E.to(gpu, non_blocking=False)
            H_prime = H_prime.to(gpu, non_blocking=False) 
            E_prime = E_prime.to(gpu, non_blocking=False)
            
            optimizer.zero_grad()
            online_pred_one, online_pred_two, target_proj_one, target_proj_two, pred_one_e, pred_one_h = model(H, E, H_prime, E_prime)
            loss_outputs, part_byol, part_recon = loss_fn(online_one=online_pred_one,
                                                          online_two=online_pred_two,
                                                          target_one=target_proj_one,
                                                          target_two=target_proj_two,
                                                          pred_one_e=pred_one_e, 
                                                          pred_one_h=pred_one_h,
                                                          h=H, e=E)

            losses.append(loss_outputs.item())
            
            loss_outputs.backward()
            optimizer.step()

            p_bar.set_description('Epoch {}'.format(epoch))
            p_bar.set_postfix(tot_loss=loss_outputs.item(), byol_loss=part_byol,
                              recon_loss=part_recon)
            if model.module.use_momentum!=False:
                model.module.update_moving_average() 

    print('Epoch: {}\ttotal_loss {:.6f}'.format(epoch, np.mean(losses)))


def eval_epoch(eval_loader, model, model_type, loss_fn, gpu, epoch, early_stopping=None):
    with torch.no_grad():
        model.eval()
        val_loss = []
        p_bar = tqdm(eval_loader)
        if model_type == 'cs':
            for H, E, _ in p_bar:
                H = H.to(gpu, non_blocking=False)
                E = E.to(gpu, non_blocking=False)
                
                pred_one_e, pred_one_h = model(H, E)
                loss_outputs, _ = loss_fn(pred_one_e=pred_one_e, pred_one_h=pred_one_h,
                                       h=H, e=E)

                val_loss.append(loss_outputs.item())

                p_bar.set_description('Epoch {}'.format(epoch))
                p_bar.set_postfix(loss=loss_outputs.item())

        elif model_type == 'cs-co':
            for H, E, H_prime, E_prime, _ in p_bar:
                H = H.to(gpu, non_blocking=False)
                E = E.to(gpu, non_blocking=False)
                H_prime = H_prime.to(gpu, non_blocking=False)
                E_prime = E_prime.to(gpu, non_blocking=False)
                
                online_pred_one, online_pred_two, target_proj_one, target_proj_two, pred_one_e, pred_one_h = model(H, E, H_prime, E_prime)
                loss_outputs, part_byol, part_recon = loss_fn(online_one=online_pred_one,
                                                              online_two=online_pred_two,
                                                              target_one=target_proj_one,
                                                              target_two=target_proj_two,
                                                              pred_one_e=pred_one_e, 
                                                              pred_one_h=pred_one_h,
                                                              h=H, e=E)
 
                val_loss.append(loss_outputs.item())
            
                p_bar.set_description('Epoch {}'.format(epoch))
                p_bar.set_postfix(tot_loss=loss_outputs.item(), byol_loss=part_byol,
                                  recon_loss=part_recon)

                
    print('val Loss {:.6f}'.format(np.mean(val_loss)))   
    #if early_stopping != None:
    #    early_stopping(np.mean(val_loss), model, epoch)
    return np.mean(val_loss)


def main(gpu, args):
    rank = args['nr'] * args['gpus'] + gpu
    dist.init_process_group('nccl', rank=rank, world_size=args['world_size'])
    torch.cuda.set_device(gpu)
    #print(gpu) 
    method_name = args['method_name']
    train_root = args['train_root']
    training_size = args['training_size']
    valid_root = args['valid_root']
    checkpoint_path = args['checkpoint_path']
    backbone_name = args['backbone_name']
    in_channel = args['in_channel']
    model_type = args['model_type']
    half_channel = args['half_channel']
    pretrained_recon = args['pretrained_recon']
    epochs = args['epochs']
    batch_size = args['batch_size']
    optim_name = args['optim_name']
    lr = args['lr']
    weight_decay = args['weight_decay']
    moving_average_decay = args['moving_average_decay']
    use_momentum = args['use_momentum']
    decoder_freeze = args['decoder_freeze']
    l1orl2 = args['l1orl2']
    training_list = args['training_list']
    gamma2 = args['gamma2']

    model = Cs_co(network=backbone_name, in_channel=in_channel, model_type=model_type,
                  half_channel=half_channel, decoder_freeze=decoder_freeze, 
                  pretrained_recon=pretrained_recon,
                  moving_average_decay=moving_average_decay,
                  use_momentum=use_momentum)
    model = model.to(gpu)
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    
    if optim_name[0] == 'S':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr, momentum=0.9,
                              weight_decay=weight_decay)
    if optim_name[0] == 'A':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr, weight_decay=weight_decay)
                    
   
    if optim_name.split('-')[-1] == 'step':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
        
    elif optim_name.split('-')[-1] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, verbose=True)

    elif optim_name.split('-')[-1] == 'no':
        scheduler = None
    
    ''' 
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1,
                                              total_epoch=5,
                                              after_scheduler=scheduler)
    '''
   

    loss_fn = Csco_Loss(model_type=model_type, l1orl2=l1orl2,
                        gamma2=gamma2) #gamma2=1.5 (resnet50)
    train_dataset = get_training_set(train_root,
                                     training_list, model_type=model_type)
    eval_dataset = get_validating_set(valid_root, model_type=model_type) 

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args['world_size'],
                                                                    rank=rank)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,
                                                                   num_replicas=args['world_size'],
                                                                   rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, num_workers=4, pin_memory=True,
                              sampler=train_sampler)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False, num_workers=4,
                             pin_memory=True, sampler=eval_sampler)

    if gpu == 0:
        early_stopping = EarlyStopping(patience=15, verbose=True,
                                       path=checkpoint_path+'{}_{}_{}_{}_{}_{}_{}_{}_{}.pth'.format(method_name,model_type,optim_name,training_size,batch_size,lr,weight_decay,moving_average_decay,gamma2))
    else:
        early_stopping = None 
    
    
    for epoch in range(epochs):
        print('epoch {}/{}'.format(epoch+1, epochs))
        train_sampler.set_epoch(epoch)
        train_epoch(train_loader, model, model_type, loss_fn, optimizer, gpu, epoch+1)
        eval_loss = eval_epoch(eval_loader, model, model_type, loss_fn, gpu, epoch+1, early_stopping)
        mean_eval_loss = torch.tensor(eval_loss/args['gpus']).to(gpu)
        dist.barrier()
        dist.all_reduce(mean_eval_loss)
        print('gpu {} eval_loss:{}, mean_loss:{}'.format(gpu, eval_loss,
                                                         mean_eval_loss.cpu().numpy()))
         
        if optim_name.split('-')[-1] == 'step':
            scheduler.step(mean_eval_loss.cpu().numpy())  
        elif optim_name.split('-')[-1] == 'cosine':
            scheduler.step()  
        elif optim_name.split('-')[-1] == 'no':
            pass
        
        if gpu == 0:
            early_stopping(mean_eval_loss.cpu().numpy(), model, epoch+1)
        '''
        if early_stopping.early_stop:
            print('Early stop!')
            break
        '''
    cleanup()



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
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # parse config file
    assert len(sys.argv)==3, 'Please claim the config file and num of GPU'
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
    args['in_channel'] = cf.getint(method_name, 'in_channel')
    args['model_type'] = cf.get(method_name, 'model_type')
    args['half_channel'] = cf.getboolean(method_name, 'half_channel')
    args['pretrained_recon'] = cf.get(method_name, 'pretrained_recon')
    args['epochs'] = cf.getint(method_name, 'epochs')
    args['batch_size'] = cf.getint(method_name, 'batch_size')
    args['optim_name'] = cf.get(method_name, 'optim')
    args['lr'] = cf.getfloat(method_name, 'lr')
    args['weight_decay'] = cf.getfloat(method_name, 'weight_decay')
    args['moving_average_decay'] = cf.getfloat(method_name, 'moving_average_decay')
    args['use_momentum'] = cf.getboolean(method_name, 'use_momentum')
    args['decoder_freeze'] = cf.getboolean(method_name, 'decoder_freeze')
    args['l1orl2'] = cf.get(method_name, 'l1orl2')
    args['gamma2'] = cf.getfloat(method_name, 'gamma2')
    

    args['training_list'] = data_list(args['train_root'],
                                      k=args['training_size'])

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    print(args['pretrained_recon'])
    print(args['gpus'])
    mp.spawn(main, args=(args,), nprocs=args['gpus'])





        


