import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                                       nn.BatchNorm1d(out_channels),)

        else:
            
            
            self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 31, 1, 15),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, 1, 1),)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, inputs):
        
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs) + inputs

        return self.relu(outputs)

class  Model_FCRN(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(Model_FCRN, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.out_channels  = out_channels
        
        filters = [16, 16, 16, 16]

        self.layer1 = modelConv1(self.in_channels, filters[0], self.is_batchnorm)
        
        self.group1 = nn.Sequential(modelResBlock(filters[0], filters[1], self.is_batchnorm),
                                    modelResBlock(filters[1], filters[2], self.is_batchnorm),
                                    modelResBlock(filters[2], filters[3], self.is_batchnorm))
        
        self.final   = nn.Sequential(nn.Conv1d(filters[3], self.out_channels, 1), 
                                     nn.ReLU(inplace=True))
        
    def forward(self, inputs):
        
        layer1 = self.layer1(inputs)
        group1 = self.group1(layer1)
        
        return self.final(group1)
