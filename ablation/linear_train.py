#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-13 13:10
# Email: yps18@mails.tsinghua.edu.cn
# Filename: linear_train.py
# Description: 
#   train the linear classifier 
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

from utils import EarlyStopping
from dataset import get_training_set, get_validating_set, get_testing_set

import sys
sys.path.append('..')
from model import Linear, Simsiam, Byol, Cs_co, Chen_mia, Xie_miccai 

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'


TRAIN_DATA = '/data/user/yangpengshuai/MedIA_SI_datasets/NCT_CRC/train/'
TEST_DATA = '/data/user/yangpengshuai/MedIA_SI_datasets/NCT_CRC/test/'

def get_img_list(path, k=None, rand_seed=8):
    total_list = os.listdir(path)
    total_list.sort()
    random.seed(rand_seed)
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


def train_test_embedding(method, backbone, total_train_list, cls, ckpt, device):
    if method == 'resnet-random':
        embedding_net = timm.create_model(backbone, pretrained=False, num_classes=0)
    elif method == 'resnet-pretrained':
        embedding_net = timm.create_model(backbone, pretrained=True, num_classes=0)
    elif method == 'byol':
        embedding_net = Byol(backbone, 224, return_embedding=True)
        embedding_state_dict = embedding_net.state_dict()
        if backbone == 'resnet18':
            pretrain_dict = torch.load('../checkpoint/NCT_CRC/resnet18/byol/byol_SGD-cosine_None_43_0.05_1e-06_0.99_91_0.07909.pth')
        elif backbone == 'resnet50':
            pretrain_dict = torch.load('../checkpoint/NCT_CRC/resnet50/byol/byol_SGD-cosine_None_43_0.05_1e-06_0.99_94_0.07873.pth')
        state_dict = {k:v for k,v in pretrain_dict.items() if
                      k in embedding_state_dict}
        embedding_net.load_state_dict(state_dict)
    elif method == 'simsiam':
        embedding_net = Simsiam(backbone, 224, return_embedding=True)
        embedding_state_dict = embedding_net.state_dict()
        if backbone == 'resnet18':
            pretrain_dict = torch.load('../checkpoint/NCT_CRC/resnet18/simsiam/simsiam_SGD-cosine_None_43_0.05_1e-06_0.99_65_0.09638.pth')
        elif backbone == 'resnet50':
            pretrain_dict = torch.load('../checkpoint/TCGA_LIHC/big_data/simsiam/simsiam_SGD-cosine_None_43_0.05_1e-06_0.99_82_0.12080.pth')
        state_dict = {k:v for k,v in pretrain_dict.items() if
                      k in embedding_state_dict}
        embedding_net.load_state_dict(state_dict)
    elif method == 'cs':
        embedding_net = Cs_co(backbone, 1, half_channel=False, return_embedding=True) 
        embedding_state_dict = embedding_net.state_dict()
        if backbone == 'resnet18':
            pretrain_dict = torch.load('../checkpoint/TCGA_LIHC/big_data/csco/cs/csco_cs_Adam-step_None_32_0.001_0.0_1.0_48_0.04527.pth')
        elif backbone == 'resnet50':
            pretrain_dict = torch.load('../checkpoint/NCT_CRC/resnet50/csco/cs/csco_cs_Adam-step_None_32_0.001_0.0_1.0_45_0.04201.pth')
        state_dict = {k:v for k,v in pretrain_dict.items() if
                      k in embedding_state_dict}
        print(len(state_dict))
        embedding_net.load_state_dict(state_dict)
    elif method == 'cs-co':
        embedding_net = Cs_co(backbone, 1, half_channel=False, return_embedding=True) 
        embedding_state_dict = embedding_net.state_dict()
        if backbone == 'resnet18':
            pretrain_dict = torch.load(ckpt)
            #pretrain_dict = torch.load('/home/yangpengshuai/old_3090_home/yps_miccai2021/datasize_co/checkpoint/10000/csco_co_Adam-step_10000_96_0.001_1e-06_1.0_21_0.07079.pth')
            #pretrain_dict = torch.load('../checkpoint/NCT_CRC/resnet18/csco/co/csco_cs-co_Adam-step_10000_32_0.001_1e-06_1.0_40_0.06224.pth')
            #pretrain_dict = torch.load('../checkpoint/TCGA_LIHC/big_data/csco/co/csco_cs-co_Adam-step_None_32_0.0001_1e-06_1.0_10.0_22_0.48475.pth')
            #pretrain_dict = torch.load('../nct_gamma_ablation/checkpoint/gamma2_10.0/csco_cs-co_Adam-step_None_32_0.001_1e-06_1.0_10.0_20_0.41737.pth')
            ########################
            #pretrain_dict = torch.load('../nct_gamma_ablation/checkpoint/old_ckpt/0.5/csco_cs-co_Adam-step_None_32_0.0001_1e-06_1.0_0.5_54_0.03911.pth')
            #pretrain_dict = torch.load('../nct_gamma_ablation/checkpoint/old_ckpt/1.0/csco_cs-co_Adam-step_None_32_0.0001_1e-06_1.0_1.0_47_0.05752.pth')
            #pretrain_dict = torch.load('../nct_gamma_ablation/checkpoint/old_ckpt/2.0/csco_cs-co_Adam-step_None_32_0.0001_1e-06_1.0_2.0_49_0.09404.pth')
            #pretrain_dict = torch.load('../nct_gamma_ablation/checkpoint/old_ckpt/5.0/csco_cs-co_Adam-step_None_32_0.0001_1e-06_1.0_5.0_57_0.20406.pth')
            #pretrain_dict = torch.load('../nct_gamma_ablation/checkpoint/old_ckpt/10.0/csco_cs-co_Adam-step_None_32_0.0001_1e-06_1.0_10.0_57_0.39169.pth')
            #########################
        elif backbone == 'resnet50':
            #pretrain_dict = torch.load('../checkpoint/NCT_CRC/resnet50/csco/co/csco_cs-co_Adam-step_10000_12_0.001_1e-06_1.0_98_0.09903.pth')
            #pretrain_dict = torch.load('./checkpoint/resnet50_10.0/csco_cs-co_Adam-step_None_12_0.0001_1e-06_1.0_10.0_52_0.38586.pth')
            pretrain_dict = torch.load('./checkpoint/resnet50_0.001_10.0/csco_cs-co_Adam-step_None_12_0.001_1e-06_1.0_10.0_26_0.39891.pth')
        state_dict = {k:v for k,v in pretrain_dict.items() if
                      k in embedding_state_dict}
        embedding_net.load_state_dict(state_dict)
    elif method == 'chen-mia':
        embedding_net = Chen_mia(backbone, 3, return_embedding=True)
        embedding_state_dict = embedding_net.state_dict()
        if backbone == 'resnet18':
            pretrain_dict = torch.load('../checkpoint/NCT_CRC/resnet18/chen/chen_mia_Adam-step_None_64_0.001_1e-08_92_0.00321.pth')
        elif backbone == 'resnet50':
            pretrain_dict = torch.load('../checkpoint/NCT_CRC/resnet50/chen/chen_mia_Adam-step_None_64_0.001_1e-08_1_0.01474.pth')
        state_dict = {k:v for k,v in pretrain_dict.items() if
                      k in embedding_state_dict}
        embedding_net.load_state_dict(state_dict)
    elif method == 'xie-miccai':
        embedding_net = Xie_miccai(backbone, 3, return_embedding=True)
        embedding_state_dict = embedding_net.state_dict()
        if backbone == 'resnet18':
            pretrain_dict = torch.load('../checkpoint/NCT_CRC/resnet18/xie/xie_miccai_SGD-step_None_64_0.001_1e-08_5_0.88814.pth')
        elif backbone == 'resnet50':
            pretrain_dict = torch.load('../checkpoint/NCT_CRC/resnet50/xie/xie_miccai_SGD-step_None_64_0.001_1e-08_2_1.20378.pth')
        state_dict = {k:v for k,v in pretrain_dict.items() if
                      k in embedding_state_dict}
        embedding_net.load_state_dict(state_dict)

    embedding_net = embedding_net.to(device)
    
    single_channel = False if method not in ['cs', 'cs-co'] else True 
    train_dataset = get_testing_set(TRAIN_DATA, cls, single_channel, total_train_list)
    test_dataset = get_testing_set(TEST_DATA, cls, single_channel)

    train_loader = DataLoader(train_dataset, batch_size=32,                 #128
                              shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,    #128
                             drop_last=False, num_workers=4)

    train_embedding, train_label = compute_embedding(train_loader,
                                                     embedding_net, device,
                                                     single_channel) 
    test_embedding, test_label = compute_embedding(test_loader, embedding_net,
                                                   device, single_channel)

    print(train_embedding.shape, train_label.shape)
    print(test_embedding.shape, test_label.shape)

    del embedding_net 
    torch.cuda.empty_cache()
    return train_embedding, train_label, test_embedding, test_label

def compute_embedding(data_loader, embedding_net, device, single_channel): 
    with torch.no_grad():
        embedding_net.eval()
        if single_channel:
            for batch_idx, (h, e, y) in enumerate(data_loader):
                h = h.to(device)
                e = e.to(device)

                tmp_embeddings = embedding_net(h,e)

                tmp_y_true = y.numpy()
                tmp_embeddings = tmp_embeddings.detach().cpu().numpy()

                if batch_idx == 0:
                    y_true = tmp_y_true
                    all_embeddings = tmp_embeddings
                else:
                    y_true = np.concatenate([y_true, tmp_y_true])
                    all_embeddings = np.concatenate([all_embeddings, tmp_embeddings])
        else:
            for batch_idx, (patches, y) in enumerate(data_loader):
                patches = patches.to(device)

                tmp_embeddings = embedding_net(patches)

                tmp_y_true = y.numpy()
                tmp_embeddings = tmp_embeddings.detach().cpu().numpy()

                if batch_idx == 0:
                    y_true = tmp_y_true
                    all_embeddings = tmp_embeddings
                else:
                    y_true = np.concatenate([y_true, tmp_y_true])
                    all_embeddings = np.concatenate([all_embeddings, tmp_embeddings])
    return all_embeddings, y_true


def train_epoch(train_loader, model, optimizer, loss_fn, device, epoch):
    model.train()
    losses = []
    total_loss = 0
    
    for batch_idx, (embed, y) in enumerate(train_loader):
        
        embed = embed.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_prob = model(embed)

        loss_outputs = loss_fn(y_prob, y)
        losses.append(loss_outputs.item())
        total_loss += loss_outputs.item()
        
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

        if batch_idx % 50 == 0:
            message = 'Train: [{}/{} ({:.2f}%)]\tLoss: {:.6f}, ACC: {:.6f}'.format(int(batch_idx*y.size(0)), len(train_loader.dataset), 100.*batch_idx/len(train_loader), np.mean(losses), np.mean(y_true==y_pred))
            print(message)
            losses = []

    print('Epoch: {}, total_loss: {:.6f}, acc: {:.6f}'.format(epoch, total_loss/(batch_idx+1), np.mean(y_true==y_pred)))


def eval_epoch(eval_loader, model, loss_fn, device, epoch, eval_type,
               early_stopping=None):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, (embed, y) in enumerate(eval_loader):
            embed = embed.to(device)
            y = y.to(device)
            
            y_prob = model(embed)

            loss_outputs = loss_fn(y_prob, y)
            val_loss += loss_outputs.item()

            tmp_y_true = y.detach().cpu().numpy()
            tmp_y_pred = torch.argmax(y_prob, dim=1).detach().cpu().numpy()
        
            if batch_idx==0:
                y_true = tmp_y_true 
                y_pred = tmp_y_pred 
            else:
                y_true = np.concatenate([y_true, tmp_y_true])
                y_pred = np.concatenate([y_pred, tmp_y_pred])

    print('{} loss: {:.6f}, acc: {:.6f}'.format(eval_type, val_loss/(batch_idx+1), np.mean(y_true==y_pred)))
    if eval_type == 'valid':
        early_stopping(val_loss/(batch_idx+1), model)
        return val_loss/(batch_idx+1)
    elif eval_type == 'test':
        return np.mean(y_true==y_pred)

            



class Narray_Dataset(Dataset):
    def __init__(self, embeddings, labels):
        super(Narray_Dataset, self).__init__()
        self.embeddings = embeddings 
        self.labels = labels

    def __getitem__(self, index):
        x = self.embeddings[index]
        y = self.labels[index]
        return torch.from_numpy(x), y

    def __len__(self):
        return len(self.labels)




if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser(description='script for linear evaluation.')
    parser.add_argument('-m', '--method', dest='method', type=str,
                        choices=['resnet-random', 'resnet-pretrained', 'byol',
                                 'simsiam', 'cs', 'cs-co', 'chen-mia', 'xie-miccai'], help='embedding method')
    parser.add_argument('-w', '--weight', dest='ckpt', type=str,
                        help='path of ckpt')

    parser.add_argument('-b', '--backbone', dest='backbone', type=str,
                        default='resnet50', choices=['resnet18', 'resnet50'], 
                        help='backbone')
    parser.add_argument('-d', '--datasize', dest='train_data_size', type=int,
                        default=1000, help='train data size')
    # get embedding method and backbone type
    args = parser.parse_args()

    # hyper-parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 500
    lr = 0.001
    train_data_size = args.train_data_size
    num_run = 10
    ratio = 0.8

    # classes and index
    #cls = {'NOR': 0, 'BEN': 1, 'INS': 2, 'INV': 3}
    cls = {'ADI':0, 'DEB':1, 'LYM':2, 'MUC':3, 'MUS':4, 'NORM':5, 'STR':6, 'TUM':7}
    print(cls)
    
    
    # linear classifier and cross-validation
    results = []
    e_losses = []
    stop_epochs = []
    for i in range(num_run):
        total_train_list = np.array(get_img_list(TRAIN_DATA+'patches/',
                                                 k=train_data_size, rand_seed=i))
        print(total_train_list[:5])
        train_data_info(total_train_list)

        # compute data embeddings  
        train_embedding, train_label, test_embedding, test_label = train_test_embedding(args.method, args.backbone, total_train_list, cls, args.ckpt, device)    
        embed_dim = train_embedding.shape[1]

        total_index = [i for i in range(len(total_train_list))]
        train_index = total_index[:int(ratio*len(total_index))]
        valid_index = total_index[int(ratio*len(total_index)):]

        model = Linear(embed_dim, len(cls))
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        #optimizer = optim.Adam(model.parameters(), lr=lr)

        train_dataset = Narray_Dataset(train_embedding[train_index],
                                       train_label[train_index])
        valid_dataset = Narray_Dataset(train_embedding[valid_index],
                                       train_label[valid_index])
        test_dataset = Narray_Dataset(test_embedding, test_label)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                                  shuffle=False, drop_last=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, drop_last=False, num_workers=4)

        loss_fn = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
        ckpt_path = './model/{}_'.format(args.method)+str(i)+'.pth'
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
    print('5-time run test_acc, mean: {:.6f}, std: {:.6f}\n'.format(results.mean(),
                                                                results.std()))

    f = open('Comparison.txt', 'a')
    f.write('backbone: {}, method: {}, train_data_size: {}\n'.format(args.backbone, args.method,
                                                                     args.train_data_size))
    f.write('{}\n'.format(results))
    f.write('test_acc, mean: {:.6f}, std: {:.6f}\n\n'.format(results.mean(),
                                                             results.std()))
    f.close()
