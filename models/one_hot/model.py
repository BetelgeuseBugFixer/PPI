import torch
import torch.nn as nn
import torch.nn.functional as F


from model import model_parts as mp

MAX_PROTEIN_LENGTH = 100
N_PFAMS = 800

PROTEIN_ALPHABET = {
    '-': 0,
    'A': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'K': 9,
    'L': 10,
    'M': 11,
    'N': 12,
    'P': 13,
    'Q': 14,
    'R': 15,
    'S': 16,
    'T': 17,
    'V': 18,
    'W': 19,
    'Y': 20,
}

# Here is the original code for the ProtENN2 model in tensorflow:
# https://github.com/iponamareva/protcnn/blob/main/layers.py

class ProtENN2_one_hot(nn.Module):
    
    def __init_(self, cnn_dim=128):
        super(ProtENN2_one_hot, self).__init__()

        # Todo: turn these into residual blocks
        # Input shape: (batch_size, MAX_PROTEIN_LENGTH, 21)
        self.res1 = mp.ResidualBlock(in_channels=21, out_channels=cnn_dim)
        self.res2 = mp.ResidualBlock(in_channels=cnn_dim, out_channels=cnn_dim)
        self.res3 = mp.ResidualBlock(in_channels=cnn_dim, out_channels=cnn_dim)
        self.res4 = mp.ResidualBlock(in_channels=cnn_dim, out_channels=cnn_dim)
        self.res5 = mp.ResidualBlock(in_channels=cnn_dim, out_channels=1100)
        # Output shape: (batch_size, 1100, MAX_PROTEIN_LENGTH)

        self.final_linear = nn.Linear(1100, N_PFAMS, activation="softmax")

    def forward(self, x):
        # Input shape: (batch_size, MAX_PROTEIN_LENGTH, 21) 

        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 21, MAX_PROTEIN_LENGTH)

        x = F.relu(self.res1(x))
        x = F.relu(self.res2(x))
        x = F.relu(self.res3(x))
        x = F.relu(self.res4(x))
        x = F.relu(self.res5(x))

        x = x.permute(0, 2, 1)  # Change shape to (batch_size, MAX_PROTEIN_LENGTH, 1100) 

        return self.final_linear(x)

