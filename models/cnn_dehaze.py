# models/cnn_dehaze.py (COMPLETELY REWRITTEN)
import torch
import torch.nn as nn
from models.attention import CBAM, ChannelAttention, SpatialAttention


class ConvBlock(nn.Module):
    """Improved Convolution Block with BN and Activation"""
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_c, out_c, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual Block with skip connection"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.attention = CBAM(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.attention(out)
        out = out + residual
        out = self.relu(out)
        return out


class EnhancedCNNDehaze(nn.Module):
    """
    Enhanced CNN with:
    - Deeper architecture (better feature extraction)
    - Batch normalization (stable training)
    - Residual connections (easier training)
    - CBAM attention (channel + spatial focus)
    - Skip connections (information flow)
    """
    def __init__(self):
        super().__init__()
        
        # Initial feature extraction
        self.initial = ConvBlock(3, 64)
        
        # Encoder: Progressive feature extraction
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        
        # Bottleneck: Deep feature processing
        self.bottleneck = ResidualBlock(64)
        
        # Decoder: Feature refinement
        self.res5 = ResidualBlock(64)
        self.res6 = ResidualBlock(64)
        
        # Feature reduction
        self.mid = ConvBlock(64, 32)
        
        # Output layer
        self.out = nn.Conv2d(32, 3, 3, padding=1)
    
    def forward(self, x):
        # Store input for residual learning
        residual_input = x
        
        # Initial extraction
        x = self.initial(x)
        
        # Deep feature processing with residuals
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoding
        x = self.res5(x)
        x = self.res6(x)
        
        # Refinement
        x = self.mid(x)
        
        # Output
        x = self.out(x)
        
        # Residual learning: predict the haze to remove
        out = residual_input + x
        
        # Clamp to valid image range [0, 1]
        out = torch.clamp(out, 0, 1)
        
        return out