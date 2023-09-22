import torch
import torch.nn as nn
import torch.nn.functional as F

class modelConv1(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(modelConv1, self).__init__()
        # Kernel size: 3, Stride: 1, Padding: 1
        if is_batchnorm:
            # 31, 1, 15
            self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, 1, 1),
                                       nn.BatchNorm1d(out_channels),
                                       nn.ReLU(inplace=True),)

        else:
            
            self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, 1, 1),
                                       nn.ReLU(inplace=True),)
            
    def forward(self, inputs):
        
        outputs = self.conv1(inputs)

        return outputs

class modelResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(modelResBlock, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            # 31, 1, 15
            self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, 1, 1),
                                       nn.BatchNorm1d(out_channels),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv1d(out_channels, out_channels, 3, 1, 1),
                                       nn.BatchNorm1d(out_channels),)

        else:
            
            # 31, 1, 15
            self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, 1, 1),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, 1, 1),)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, inputs):
        
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs) + inputs

        return self.relu(outputs)
    
class  CLFCRN_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(CLFCRN_Encoder, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.out_channels  = out_channels
        
        filters = [16, 16, 16, 16]
        
        self.conv = nn.Sequential(
                                  modelConv1(self.in_channels, filters[0], self.is_batchnorm),
                                  modelResBlock(filters[0], filters[1], self.is_batchnorm),
                                  modelResBlock(filters[1], filters[2], self.is_batchnorm),
                                  modelResBlock(filters[2], filters[3], self.is_batchnorm),
                                  nn.Conv1d(filters[3], self.out_channels, 3, 1, 1),
                                  nn.ReLU(inplace=True)
        )
        
    def forward(self, inputs):
            
        return self.conv(inputs)

    
class  CLFCRN_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(CLFCRN_Decoder, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.out_channels  = out_channels
        
        filters = [16, 16, 16, 16]
        
        self.deconv = nn.Sequential(modelConv1(self.in_channels, filters[0], self.is_batchnorm),
                                    modelResBlock(filters[0], filters[1], self.is_batchnorm),
                                    modelResBlock(filters[1], filters[2], self.is_batchnorm),
                                    modelResBlock(filters[2], filters[3], self.is_batchnorm),
                                    nn.Conv1d(filters[3], self.out_channels, 3, 1, 1),
                                    nn.ReLU(inplace=True))
        
    def forward(self, inputs):
             
        return self.deconv(inputs)
    
    
class  Network_clFCRN(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(Network_clFCRN, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.out_channels  = out_channels
        
        self.encoder = CLFCRN_Encoder(in_channels, out_channels, is_batchnorm)

        self.decoder = CLFCRN_Decoder(in_channels, out_channels, is_batchnorm)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, seismic_data, seismic_data_pre_v, label_velocity):
        
#         pre_velocity = self.encode(seismic_data)
#         recon_seismic_data = self.decode(self.encode(seismic_data))
        
        pre_velocity = self.encode(seismic_data_pre_v)
        recon_seismic_data_pre_v = self.decode(pre_velocity)
        
        pre_seismic_data = self.decode(label_velocity)
        recon_velocity = self.encode(pre_seismic_data)
        
        return self.decode(self.encode(seismic_data)), pre_velocity, recon_seismic_data_pre_v, pre_seismic_data, recon_velocity
            
    
    
    
