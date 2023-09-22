# -*- coding: utf-8 -*-
"""
Created on June 2023

@author: Yuping Wu (ypwu@stu.hit.edu.cn)

"""

################################################
########        DESIGN   NETWORK        ########
################################################


import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set default dtype to float32
torch.set_default_dtype(torch.float32)

# PyTorch random number generator
torch.manual_seed(1234)

class modelConv1(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(modelConv1, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            
            self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 31, 1, 15),
                                       nn.BatchNorm1d(out_channels),
                                       nn.ReLU(inplace=True),)

        else:
            
            self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 31, 1, 15),
                                       nn.ReLU(inplace=True),)
            
    def forward(self, inputs):
        
        outputs = self.conv1(inputs)

        return outputs

class modelResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(modelResBlock, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            
            self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 31, 1, 15),
                                       nn.BatchNorm1d(out_channels),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv1d(out_channels, out_channels, 3, 1, 1),
                                       nn.BatchNorm1d(out_channels),
                                       nn.ReLU(inplace=True),)

        else:
            
            
            self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 31, 1, 15),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, 1, 1),
                                       nn.ReLU(inplace=True),)
            
    def forward(self, inputs):
        
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs) + inputs

        return outputs

class  ImpedanceModel(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(ImpedanceModel, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.out_channels  = out_channels
        
        filters = [16, 16, 16, 16, 16]

        self.layer1 = modelConv1(self.in_channels, filters[0], self.is_batchnorm)
        
        self.group1 = nn.Sequential(modelConv1(filters[0], filters[1], self.is_batchnorm),
                                    modelResBlock(filters[1], filters[1], self.is_batchnorm))
        
        self.group2 = nn.Sequential(modelConv1(filters[1], filters[2], self.is_batchnorm),
                                    modelResBlock(filters[2], filters[2], self.is_batchnorm))
        
        self.group3 = nn.Sequential(modelConv1(filters[2], filters[3], self.is_batchnorm),
                                    modelResBlock(filters[3], filters[3], self.is_batchnorm))
        
        self.group4 = nn.Sequential(modelConv1(filters[3], filters[4], self.is_batchnorm),
                                    modelResBlock(filters[4], filters[4], self.is_batchnorm))
        
        self.final   = nn.Sequential(nn.Conv1d(filters[4], self.out_channels, 1), 
                                     nn.ReLU(inplace=True))
        
    def forward(self, inputs):
        
        layer1 = self.layer1(inputs)
        group1 = self.group1(layer1)
        group2 = self.group2(group1)
        group3 = self.group3(group2)
        group4 = self.group4(group3)
        
        return self.final(group4)




