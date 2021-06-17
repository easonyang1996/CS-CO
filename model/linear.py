#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-12 11:10
# Email: yps18@mails.tsinghua.edu.cn
# Filename: linear.py
# Description: 
#       linear classifier for linear evaluation 
# ******************************************************
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, input_dim, num_cls=8):
        super(Linear, self).__init__()
        self.fc = nn.Linear(input_dim, num_cls)

    def forward(self, x):
        out = self.fc(x)
        return out
