"""DenseNet for segmentation map classification.

Adapted for 1-channel (grayscale) 256x256 input.
Output: sigmoid P(real).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.block(x)], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class DenseNetSegm(nn.Module):
    """Small DenseNet for 1-channel 256x256 segmentation maps."""

    def __init__(self, growth_rate=16, n_layers=4):
        super().__init__()
        # Stem: 1-channel input
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )  # 256 -> 64

        ch = 32
        self.block1 = DenseBlock(ch, growth_rate, n_layers)
        ch += growth_rate * n_layers
        self.trans1 = Transition(ch, ch // 2)
        ch = ch // 2

        self.block2 = DenseBlock(ch, growth_rate, n_layers)
        ch += growth_rate * n_layers
        self.trans2 = Transition(ch, ch // 2)
        ch = ch // 2

        self.block3 = DenseBlock(ch, growth_rate, n_layers)
        ch += growth_rate * n_layers

        self.bn = nn.BatchNorm2d(ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, 1)
        self._out_ch = ch

    def forward(self, x):
        x = self.stem(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.block3(x)
        x = F.relu(self.bn(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return torch.sigmoid(self.fc(x))
