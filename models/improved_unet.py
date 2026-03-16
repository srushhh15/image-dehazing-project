import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wavelet import dwt


# ----------------------------
# Channel Attention
# ----------------------------
class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.size()

        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y


# ----------------------------
# Double Convolution Block
# ----------------------------
class DoubleConv(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Improved U-Net
# ----------------------------
class ImprovedUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.d1 = DoubleConv(3, 64)
        self.att1 = ChannelAttention(64)

        self.d2 = DoubleConv(64, 128)
        self.att2 = ChannelAttention(128)

        self.d3 = DoubleConv(128, 256)
        self.att3 = ChannelAttention(256)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.u1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c1 = DoubleConv(256, 128)

        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c2 = DoubleConv(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x):

        input_img = x

        # ----------------------------
        # Wavelet Frequency Branch
        # ----------------------------
        LL, LH, HL, HH = dwt(x)

        freq = LL + LH + HL + HH

        freq = F.interpolate(
            freq,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        # Fuse spatial + frequency
        x = x + freq

        # ----------------------------
        # Encoder
        # ----------------------------
        x1 = self.att1(self.d1(x))

        x2 = self.pool(x1)
        x2 = self.att2(self.d2(x2))

        x3 = self.pool(x2)
        x3 = self.att3(self.d3(x3))

        # ----------------------------
        # Decoder
        # ----------------------------
        x = self.u1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.c1(x)

        x = self.u2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.c2(x)

        residual = self.out(x)

        # ----------------------------
        # Residual Learning
        # ----------------------------
        output = input_img + residual

        return output