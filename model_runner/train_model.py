import torch
import torch.optim as optim
import torch.nn as nn
from model.model import TransMS2Predictor
from model.dataset import ProteomeToolsDataset
from torch.utils.data import DataLoader

from dotenv import load_dotenv
import os
import wandb
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Initialize Weights & Biases (WandB) for experiment tracking
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(
    project="ms2_pytorch",
    entity='elizabeth-lochert-flx'
)


# Check for GPU availability and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model and move it to the appropriate device
model = TransMS2Predictor().to(device)

# Define the loss function and optimizer
loss_cos = nn.CosineSimilarity(dim=2, eps=1e-6)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define batch size
batch_size = 1024


# Paths to dataset files
paths = (
    # "/cmnfs/proj/prosit_astral/datasets/proteome_tools_dlomix_format_train.parquet",
    "/cmnfs/proj/prosit_astral/datasets/proteome_tools_dlomix_format_val.parquet",
    # "/cmnfs/proj/prosit_astral/datasets/proteome_tools_dlomix_format_test.parquet"
)

# Load data
print("Loading data")

dataset = ProteomeToolsDataset(paths)

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Data loaded successfully")


# Training loop parameters
num_epochs = 10
total_ticks = 20  # Number or ticks

print("Starting training...")
print(f"Total batches per epoch: {len(train_loader)}")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("_"*total_ticks)

    for batch_idx, batch in enumerate(train_loader):

        # Move batch data to the gpu (ideallly)
        x_sequence = batch[0].to(device)
        x_metadata = batch[1].to(device)
        y = batch[2][:, None, :].to(device)

        # Forward pass
        y_pred = model(x_sequence, x_metadata)
        loss = loss_cos(y_pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        # Log training loss to WandB
        wandb.log({"loss": loss.mean().item()})

        # Print progress

        # Print training progress every few batches
        if batch_idx % (len(train_loader) // total_ticks + 1) == 0:
            print("=", end="")


    # Print epoch loss
    print(f"\nLoss: {loss.mean().item()}\n\n")
