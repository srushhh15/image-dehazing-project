"""
CNN-based Image Dehazing Model
Inspired by AOD-Net (All-in-One Dehazing Network)
Uses the atmospheric scattering model: I(x) = J(x)*t(x) + A*(1-t(x))
Solves for a unified transformation K(x) to recover clean image J(x).
"""

import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    """Basic Conv + BatchNorm + ReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class CNNDehaze(nn.Module):
    """
    CNN Dehazing Network.

    Architecture:
      - K1: 1x1 conv on input (3 ch -> 3 ch)
      - K2: 3x3 conv on input (3 ch -> 3 ch)
      - K3: 5x5 conv on concat(K1, K2) (6 ch -> 3 ch)
      - K4: 7x7 conv on concat(K2, K3) (6 ch -> 3 ch)
      - K5: 3x3 conv on concat(K1, K4) (6 ch -> 3 ch)  -- final K(x)

    Output: J = K(x) * I - K(x) + 1  (clean image)
    """

    def __init__(self):
        super().__init__()

        self.k1 = ConvBnRelu(3, 3, kernel_size=1, padding=0)
        self.k2 = ConvBnRelu(3, 3, kernel_size=3, padding=1)
        self.k3 = ConvBnRelu(6, 3, kernel_size=5, padding=2)
        self.k4 = ConvBnRelu(6, 3, kernel_size=7, padding=3)
        self.k5 = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, padding=1),
            nn.Sigmoid()   # K(x) in [0,1]
        )

    def forward(self, x):
        k1 = self.k1(x)
        k2 = self.k2(x)
        k3 = self.k3(torch.cat([k1, k2], dim=1))
        k4 = self.k4(torch.cat([k2, k3], dim=1))
        k  = self.k5(torch.cat([k1, k4], dim=1))

        # Atmospheric scattering model inversion
        # J(x) = K(x) * I(x) - K(x) + 1
        out = k * x - k + 1
        out = torch.clamp(out, 0.0, 1.0)
        return out


# -----------------------------------------------------------------
# Enhanced CNN with residual refinement (deeper variant)
# -----------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block for the enhanced variant."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class EnhancedCNNDehaze(nn.Module):
    """
    Enhanced CNN Dehazing Network with residual refinement.

    Stages:
      1. Feature extraction  (3 -> 64 channels)
      2. Residual refinement (3 residual blocks)
      3. Transmission map estimation (64 -> 3)
      4. Image reconstruction via scattering model inversion
    """

    def __init__(self, num_residual_blocks=3):
        super().__init__()

        # Feature extraction
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Residual refinement
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        # Transmission map head
        self.transmission_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )

        # Atmospheric light estimation head (global)
        self.atm_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.feature_extract(x)
        feat = self.residual_blocks(feat)

        t = self.transmission_head(feat)          # (B,3,H,W) in [0,1]
        A = self.atm_head(feat)                   # (B,3)
        A = A.unsqueeze(-1).unsqueeze(-1)         # (B,3,1,1) for broadcast

        # Dehazing formula: J = (I - A) / max(t, 0.1) + A
        t_clamped = torch.clamp(t, min=0.1)
        out = (x - A) / t_clamped + A
        out = torch.clamp(out, 0.0, 1.0)
        return out
