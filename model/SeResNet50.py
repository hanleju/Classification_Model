import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import math
from torchvision.models import ResNet

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y

class SEResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, reduction_ratio=16):
        super(SEResNetBlock, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.se_block = SEBlock(out_channels, reduction_ratio)

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.se_block(x)
        return x
    
class SEResNet(nn.Module):
    def __init__(self, num_classes, block, layers, reduction_ratio=16):
        super(SEResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1, reduction_ratio=reduction_ratio)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2, reduction_ratio=reduction_ratio)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2, reduction_ratio=reduction_ratio)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2, reduction_ratio=reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride, reduction_ratio):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, reduction_ratio))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1, reduction_ratio=reduction_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def seresnet50(num_classes=10, reduction_ratio=16):
    return SEResNet(num_classes, SEResNetBlock, [3, 4, 6, 3], reduction_ratio)