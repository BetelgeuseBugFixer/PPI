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


class ProtENN2_one_hot(nn.Module):
    
    def __init_(self, cnn_dim=128):
        super(ProtENN2_one_hot, self).__init__()

        # Todo: turn these into residual blocks
        # Input shape: (batch_size, MAX_PROTEIN_LENGTH, 21)
        self.cnn1 = nn.Conv1d(in_channels=21, out_channels=cnn_dim, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=3, padding=1)
        self.cnn3 = nn.Conv1d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=3, padding=1)
        self.cnn4 = nn.Conv1d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=3, padding=1)
        self.cnn5 = nn.Conv1d(in_channels=cnn_dim, out_channels=1100, kernel_size=3, padding=1)
        # Output shape: (batch_size, cnn_dim, MAX_PROTEIN_LENGTH)

        self.final_dense = nn.Linear(1100, N_PFAMS)

    def forward(self, x):
        # Input shape: (batch_size, MAX_PROTEIN_LENGTH, 21) 

        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 21, MAX_PROTEIN_LENGTH)

        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = F.relu(self.cnn4(x))
        x = F.relu(self.cnn5(x))

        x = x.permute(0, 2, 1)  # Change shape to (batch_size, MAX_PROTEIN_LENGTH, 1100) 

        x = F.sigmoid(self.final_dense(x))

        return x
