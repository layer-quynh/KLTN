import torch
import torch.nn as nn

class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(conv_bn_relu, self).__init__()
        self.convBlock == nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convBlock(x)