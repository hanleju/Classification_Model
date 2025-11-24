import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate=32, reduction=0.5, num_classes=10, init_weights=True):
        super().__init__()
        self.growth_rate = growth_rate

        # Initial convolution
        num_channels = 2 * growth_rate
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense blocks
        self.dense1 = DenseBlock(num_blocks[0], num_channels, growth_rate)
        num_channels += num_blocks[0] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans1 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense2 = DenseBlock(num_blocks[1], num_channels, growth_rate)
        num_channels += num_blocks[1] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans2 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense3 = DenseBlock(num_blocks[2], num_channels, growth_rate)
        num_channels += num_blocks[2] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans3 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense4 = DenseBlock(num_blocks[3], num_channels, growth_rate)
        num_channels += num_blocks[3] * growth_rate

        # Final batch norm
        self.bn = nn.BatchNorm2d(num_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

        # Weight initialization
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        with profiler.record_function("Conv Forward"):
            out = self.conv1(x)
            out = self.trans1(self.dense1(out))
            out = self.trans2(self.dense2(out))
            out = self.trans3(self.dense3(out))
            out = self.dense4(out)
            out = F.relu(self.bn(out))
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        return out

    # Define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def DenseNet121(num_classes=10):
    return DenseNet([6, 12, 24, 16], growth_rate=32, num_classes=num_classes)
