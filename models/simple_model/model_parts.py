import torch
import torch.nn as nn

CUSTOM_ALPHABET = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19,
    'X': 20,  # unknown amino acid
    # '-': 21,
}


class ResidualBlock(nn.Module):

    def __init__(self, res_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()

        assert kernel_size % 2 == 1     # only allow odd kernel sizes to ensure padding='same' works

        self.conv1 = nn.Conv1d(res_channels, res_channels, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(res_channels, res_channels, kernel_size=kernel_size, padding='same')

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

        # print(out.shape)
        # print(identity.shape

        out += identity # Skip connection for residual learning

        return self.relu(out)

