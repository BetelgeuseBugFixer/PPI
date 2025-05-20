import torch.nn as nn

from models.simple_model import model_parts as mp

MAX_PROTEIN_LENGTH = 100

# Here is the original code for the ProtENN2 model in tensorflow:
# https://github.com/iponamareva/protcnn/blob/main/layers.py

class ProtENN2_style(nn.Module):
    
    def __init__(self, cnn_dim=128, in_channels=21, num_pfams=100):    # either 21 for one-hot input or 1024 for ProtT5 input
        super().__init__()
        
        # Input shape: (batch_size, MAX_PROTEIN_LENGTH, 21)
        self.cnn_in = nn.Conv1d(in_channels=in_channels, out_channels=cnn_dim, kernel_size=3, padding=1)

        self.res1 = mp.ResidualBlock(res_channels=cnn_dim)
        self.res2 = mp.ResidualBlock(res_channels=cnn_dim)
        self.res3 = mp.ResidualBlock(res_channels=cnn_dim)
        self.res4 = mp.ResidualBlock(res_channels=cnn_dim)

        self.cnn_out = nn.Conv1d(in_channels=cnn_dim, out_channels=1100, kernel_size=3, padding=1)
        # Output shape: (batch_size, 1100, MAX_PROTEIN_LENGTH)

        self.final_linear = nn.Linear(1100, num_pfams)
        # self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input shape: (batch_size, MAX_PROTEIN_LENGTH, 21) 

        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 21, MAX_PROTEIN_LENGTH)

        x = self.relu(self.cnn_in(x))   # required to have consistent input and output shapes for resnet

        x = self.relu(self.res1(x))
        x = self.relu(self.res2(x))
        x = self.relu(self.res3(x))
        x = self.relu(self.res4(x))

        x = self.relu(self.cnn_out(x))  # required to have proper output shape (resnet input and output shapes must always be the same)

        x = x.permute(0, 2, 1)  # Change shape to (batch_size, MAX_PROTEIN_LENGTH, 1100) 
        
        x = self.final_linear(x)
        
        return x
        # return self.softmax(x)    # not needed when using CrossEntropyLoss
