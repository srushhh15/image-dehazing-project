# models/attention.py (UPDATED & EXPANDED)
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """Channel Attention - what features to focus on"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # NEW: Also use max pooling
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        
        # Combine both
        out = avg_out + max_out
        out = self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)
        
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention - where to focus in the image"""
    def __init__(self, kernel_size=7):
        super().__init__()
        
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        out = self.conv(x_cat)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x