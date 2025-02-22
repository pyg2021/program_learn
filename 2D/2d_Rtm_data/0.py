import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """基础卷积模块（包含3D/2D自适应）"""
    def __init__(self, in_channels, out_channels, dim=3):
        super().__init__()
        if dim == 3:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:  # 2D模式
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        return self.conv(x)

class AttentionGate3D(nn.Module):
    """3D时空注意力门控机制"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
    def forward(self, g, x):
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)
        psi = F.relu(g_conv + x_conv)
        alpha = self.psi(psi)
        return alpha * x

class UpBlock(nn.Module):
    """上采样模块（含注意力门控）"""
    def __init__(self, in_channels, out_channels, dim=3, bilinear=True):
        super().__init__()
        if bilinear:
            if dim ==3:
                self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            else:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            if dim ==3:
                self.up = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=2, stride=2)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        
        self.att = AttentionGate3D(F_g=in_channels//2, F_l=in_channels//2, F_int=in_channels//4)
        self.conv = ConvBlock(in_channels, out_channels, dim=dim)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.att(x1, x2)  # 应用注意力门控
        x = torch.cat([x2, x1], dim=1)  # 通道维度拼接
        return self.conv(x)

class STANet(nn.Module):
    """Spatio-Temporal Attention U-Net for Seismic Prediction"""
    def __init__(self, in_channels=1, out_channels=1, dim=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.dim = dim
        self.pool = nn.MaxPool3d(2) if dim==3 else nn.MaxPool2d(2)
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, features[0], dim)
        self.enc2 = ConvBlock(features[0], features[1], dim)
        self.enc3 = ConvBlock(features[1], features[2], dim)
        self.enc4 = ConvBlock(features[2], features[3], dim)
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3]*2, dim)
        
        # Decoder
        self.up1 = UpBlock(features[3]*2, features[3], dim)
        self.up2 = UpBlock(features[3], features[2], dim)
        self.up3 = UpBlock(features[2], features[1], dim)
        self.up4 = UpBlock(features[1], features[0], dim)
        
        # 输出层（自适应2D/3D）
        if dim ==3:
            self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        else:
            self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        bn = self.bottleneck(self.pool(e4))
        
        # Decoder
        d1 = self.up1(bn, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)
        
        return torch.sigmoid(self.final_conv(d4))  # 输出归一化

if __name__ == "__main__":
    # 测试3D版本
    model_3d = STANet(in_channels=1, out_channels=1, dim=3)
    x = torch.randn(2, 1, 64, 64, 64)  # (batch, channel, depth, height, width)
    print("3D Output shape:", model_3d(x).shape)  # 应保持输入尺寸
    
    # 测试2D版本
    model_2d = STANet(dim=2)
    x = torch.randn(2, 1, 256, 256)
    print("2D Output shape:", model_2d(x).shape)
