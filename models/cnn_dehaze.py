import torch
import torch.nn as nn
from models.attention import ChannelAttention


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EnhancedCNNDehaze(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = ConvBlock(3, 64)
        self.a1 = ChannelAttention(64)

        self.c2 = ConvBlock(64, 64)
        self.a2 = ChannelAttention(64)

        self.c3 = ConvBlock(64, 32)

        self.out = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):

        identity = x

        x = self.c1(x)
        x = self.a1(x)

        x = self.c2(x)
        x = self.a2(x)

        x = self.c3(x)

        x = self.out(x)

        out = identity + x

        out = torch.clamp(out, 0, 1)

        return out