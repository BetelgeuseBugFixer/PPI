import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, res_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(res_channels, res_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(res_channels, res_channels, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(res_channels)
        self.bn2 = nn.BatchNorm1d(res_channels)

        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        print(out.shape)
        print(identity.shape)

        out += identity # Skip connection for residual learning

        return self.relu(out)

