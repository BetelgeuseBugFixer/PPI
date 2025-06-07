import torch
import torch.optim as optim
import torch.nn as nn

from simple_model.model import ProtENN2_style
from simple_model.model_parts import CUSTOM_ALPHABET

from dotenv import load_dotenv
import os
import wandb
import pandas as pd
import numpy as np

# Load environment variables from .env file
load_dotenv()

MAX_PROTEIN_LENGTH = 500

# Initialize Weights & Biases (WandB) for experiment tracking
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(
    project="pp1-ProtENN2",
    entity='elizabeth-lochert-flx'
)

# Check for GPU availability and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")






print("Loading training data...")

# Path must be a pkl file with the following columns: 'sequence':str, 'pfam_tensor':list<str> 
path = "../dataset/dataset.pkl"

# Load the training data
data =  pd.read_pickle(path)


# Filter out sequences longer than MAX_PROTEIN_LENGTH
data = data[data["sequence"].apply(lambda x: len(x) <= MAX_PROTEIN_LENGTH)]

# Todo
# Take small subset of data (REMOVE WHEN FINAl DATASET IS READY)
data = data.sample(n=10_000, random_state=42).reset_index(drop=True)

print("One-hot encoding sequences and converting pfams to indices...")

# One hot encode the sequences
def one_hot_encode_sequence(sequence, alphabet=CUSTOM_ALPHABET, max_length=MAX_PROTEIN_LENGTH):
    one_hot = np.zeros((max_length, len(alphabet)), dtype=np.float32)
    for i, char in enumerate(sequence):
        if i < max_length and char in alphabet:
            one_hot[i, alphabet[char]] = 1.0
    return one_hot

data["sequence_oh"] = data["sequence"].apply(one_hot_encode_sequence)

# Generate pfam to index mapping
pfam_to_index = {pfam: idx+1 for idx, pfam in enumerate(data["pfam_tensor"].explode().unique())}

# Convert pfams to indices and pad to max length
def convert_pfams_to_indices(pfams, pfam_to_index, max_length=MAX_PROTEIN_LENGTH):
    indices = [pfam_to_index[pfam] for pfam in pfams if pfam in pfam_to_index]
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))  # Pad with zeros
    return np.array(indices[:max_length], dtype=np.int64)

data["pfams_indices"] = data["pfam_tensor"].apply(lambda x: convert_pfams_to_indices(x, pfam_to_index))

print("Data preprocessing complete.")



X = torch.tensor(np.stack(data["sequence_oh"].values), dtype=torch.float32)
Y = torch.tensor(np.stack(data["pfams_indices"].values), dtype=torch.int64)

# Create a PyTorch dataset
dataset = torch.utils.data.TensorDataset(X, Y)

# Define the model and move it to the appropriate device
model = ProtENN2_style(cnn_dim=512, num_pfams=len(pfam_to_index)+1).to(device)

# Define the loss function and optimizer
loss_cel = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Validation and training loop parameters
num_epochs = 10
total_ticks = 20  # Number of ticks for progress bar

batch_size = 64  # Define the batch size

# split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("_"*total_ticks)

    model.train()  # Set the model to training mode

    for batch_idx, (x_sequence, y) in enumerate(train_loader):

        # Move batch data to the gpu (ideally)
        x_sequence = x_sequence.to(device)
        y = y.to(device)

        # Forward pass
        y_pred = model(x_sequence)
        # print(y_pred.shape, y.shape)
        loss = loss_cel(y_pred.view(-1, len(pfam_to_index)+1), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training loss to WandB
        wandb.log({"loss": loss.item()})

        # Print progress
        if batch_idx % (len(train_loader) // total_ticks + 1) == 0:
            print("=", end="")

    print(f"\nLoss: {loss.item()}")

    # Print validation loss
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for x_sequence, y in val_loader:
            x_sequence = x_sequence.to(device)
            y = y.to(device)

            y_pred = model(x_sequence)
            loss = loss_cel(y_pred.view(-1, len(pfam_to_index)+1), y.view(-1))
            val_loss += loss.item()
    val_loss /= len(val_loader)
    wandb.log({"val_loss": val_loss})
    print(f"Validation Loss: {val_loss}\n")




