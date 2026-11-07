"""Small residual network for CIFAR-10."""

import torch
from torch import nn

# WIDTHS = (48, 96, 192, 320)  # 4,947,770 trainable parameters, below the 5M limit.
# BLOCKS = (2, 2, 2, 2)

WIDTHS = (64, 128, 256)
BLOCKS = (3, 3, 3)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Identity() if stride == 1 and in_channels == out_channels else
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                          nn.BatchNorm2d(out_channels))
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class SmallResNet(nn.Module):
    def __init__(self, widths=WIDTHS, blocks=BLOCKS):
        super().__init__()
        if len(widths) != len(blocks) or not widths or any(count < 1 for count in blocks):
            raise ValueError("widths and blocks must have equal, nonzero lengths")
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        stages = []
        in_channels = widths[0]
        for i, (width, count) in enumerate(zip(widths, blocks)):
            layers = []
            for j in range(count):
                stride = 2 if i > 0 and j == 0 else 1
                layers.append(BasicBlock(in_channels, width, stride))
                in_channels = width
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1], 10)

    def forward(self, x):
        return self.fc(self.pool(self.stages(self.stem(x))).flatten(1))
