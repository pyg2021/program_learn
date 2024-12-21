#新的合成数据训练instance+128
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from DataLoad import DataLoad
from DataLoad1210 import DataLoad as DataLoad1
from Model_2DUnet1208 import net
import os 
from skimage.metrics import structural_similarity as ssim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
start=time.time()

##data_prepare
BatchSize=3

device="cuda"

model=net(2,1,64).to(device)
model=nn.parallel.DataParallel(model)

model.load_state_dict(torch.load("/home/pengyaoguang/data/2D_data/2D_result/modeltest9_8.pkl"))
model.eval()
# 定义一个包装器（wrapper），用于在卷积层上应用LORA更新
class LORAConv2d(nn.Module):
    def __init__(self, original_conv, rank, downsample_factor=16):
        super(LORAConv2d, self).__init__()
        self.original_conv = original_conv
        self.rank = rank
        self.downsample_factor = downsample_factor
        
        # 初始化LORA参数
        self.A = nn.Parameter(torch.randn(original_conv.weight.size(0), rank) / downsample_factor)
        self.B = nn.Parameter(torch.randn(rank, original_conv.weight.size(1)) / downsample_factor)
        self.s = nn.Parameter(torch.zeros(1))  # 可选的缩放参数
 
    def forward(self, x):
        # 计算低秩更新项
        delta_weight = torch.matmul(self.A, self.B) * torch.sigmoid(self.s)
        
        # 注意：这里我们不能直接修改self.original_conv.weight，因为nn.Parameter是不可变的。
        # 相反，我们会在每次前向传播时创建一个新的卷积层，使用更新后的权重。
        
        # 创建一个新的卷积层，使用原始权重加上LORA更新项
        updated_weight = self.original_conv.weight + delta_weight
        updated_conv = nn.Conv2d(
            in_channels=self.original_conv.in_channels,
            out_channels=self.original_conv.out_channels,
            kernel_size=self.original_conv.kernel_size,
            stride=self.original_conv.stride,
            padding=self.original_conv.padding,
            dilation=self.original_conv.dilation,
            groups=self.original_conv.groups,
            bias=self.original_conv.bias is not None,
            padding_mode=self.original_conv.padding_mode
        )
        updated_conv.weight = nn.Parameter(updated_weight)
        if self.original_conv.bias is not None:
            updated_conv.bias = self.original_conv.bias
        
        # 使用更新后的卷积层进行前向传播
        return updated_conv(x)
 
# 包装U-Net中的卷积层以应用LORA
def apply_lora_to_unet(model, rank, downsample_factor=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 替换卷积层为LORAConv2d包装器
            setattr(model, name, LORAConv2d(module, rank, downsample_factor))
 
# 应用LORA到预训练的U-Net模型
apply_lora_to_unet(model, rank=8)
 
# 现在，pretrained_model中的卷积层已经被LORAConv2d包装器替换，
# 在前向传播时会动态地应用LORA更新。
 
# 定义损失函数和优化器（注意：这里我们不会优化原始权重，只会优化LORA参数）
optimizer = torch.optim.Adam([
    {'params': [param for name, param in model.named_parameters() if 'A' not in name and 'B' not in name and 's' not in name], 'lr': 0.0},  # 冻结原始权重
    {'params': [param for name, param in model.named_parameters() if 'A' in name or 'B' in name or 's' in name], 'lr': 1e-4}  # 优化LORA参数
], lr=1e-4)  # 注意：这里的lr=1e-4实际上只用于LORA参数，因为原始权重被冻结了
 
# 训练循环（示例）
# ... (加载数据、前向传播、计算损失、反向传播、更新参数等)
 
# 注意：在实际的训练循环中，你需要确保只更新LORA参数（A, B, s），而保持原始权重不变。
# 这可以通过在优化器中为不同的参数组设置不同的学习率来实现，如上所示。
# 另外，由于我们每次前向传播时都会创建一个新的卷积层，这可能会增加一些计算开销。
# 因此，在实际应用中，你可能需要进一步优化这个过程，例如通过缓存更新后的权重或使用其他技巧来减少重复计算。

scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=500,gamma=0.7)
loss_1=torch.nn.L1Loss()
# loss_1=torch.nn.MSELoss()
# train(model,train_loader_1,test_loader_1,10000,device,optimizer,scheduler,loss_1,save_number=9)
# test(model,train_loader_1,loss_1,device)
# test(model,train_loader_2,loss_1,device)

# for name, module in model.named_modules():
#     if hasattr(module, 'weight'):  # 检查层是否有权重属性
#         print(f"Layer name: {name}")
#         print(f"Weights: {module.weight}\n")
#         # 如果层还有偏置项，也可以打印出来
#         if hasattr(module, 'bias'):
#             print(f"Biases: {module.bias}\n")
