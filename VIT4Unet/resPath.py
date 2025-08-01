import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBN(nn.Module):
    """ conv2d + batch norm 组合层 """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation == 'relu':
            x = F.relu(x)
        return x

class ResPath(nn.Module):
    """ PyTorch 版本的 ResPath """
    def __init__(self, filters, length, in_channels):
        super(ResPath, self).__init__()
        self.length = length
        self.filters = filters
        
        # 创建第一个残差块
        self.initial_shortcut = ConvBN(in_channels, filters, 1, padding='same', activation=None)
        self.initial_conv = ConvBN(in_channels, filters, 3, padding='same', activation='relu')
        
        # 创建后续的残差块
        self.blocks = nn.ModuleList()
        for _ in range(length-1):
            block = nn.ModuleDict({
                'shortcut': ConvBN(filters, filters, 1, padding='same', activation=None),
                'conv': ConvBN(filters, filters, 3, padding='same', activation='relu')
            })
            self.blocks.append(block)
    
    def forward(self, x):
        # 第一个残差块
        shortcut = self.initial_shortcut(x)
        out = self.initial_conv(x)
        out = out + shortcut
        out = F.relu(out)
        
        # 后续残差块
        for i in range(self.length-1):
            shortcut = self.blocks[i]['shortcut'](out)
            out = self.blocks[i]['conv'](out)
            out = out + shortcut
            out = F.relu(out)
        
        return out