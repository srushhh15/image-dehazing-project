import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.u2 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))

        x = self.u1(x2)
        x = self.u2(torch.cat([x, x1], dim=1))

        return self.out(x)