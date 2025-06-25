import torch
import torch.optim as optim
import torch.nn as nn

from simple_model.model import ProtENN2_style

from dotenv import load_dotenv
import os
import wandb
import pandas as pd
import numpy as np
import h5py
import yaml
import json

# Load environment variables from .env file
load_dotenv()

# Print Run Information
print('='*32)
print('Conda info')
print(f"Environment: {os.environ['CONDA_DEFAULT_ENV']}")
print('='*32)
print('PyTorch info')
print("PyTorch version:", torch.__version__)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")
print(f"List of GPUs available: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
print('='*32)

# Check for GPU availability and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load config file and print settings
with open("./config.yaml", 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

dataset_settings = config["dataset_settings"]
model_settings = config['model_settings']
train_settings = config['train_settings']

print("\nDataset Settings:")
for key, value in dataset_settings.items():
    print(f"{key}: {value}")

if config['model_type'] == 'oh':
    print("\nOne-hot input Model config:")
    for key, value in model_settings.items():
        print(f"{key}: {value}")
elif config['model_type'] == 'emb':
    print("\nProtT5 input Model config:")
    for key, value in model_settings.items():
        print(f"{key}: {value}")

print("\nTraining Settings:")
for key, value in train_settings.items():
    print(f"{key}: {value}")
print('='*32)


# Set constants from config
MAX_PROTEIN_LENGTH = dataset_settings['max_protein_length']

##########################################################################################################
##########################################################################################################

print("Loading training data...")

# Select dataset path based on dataset_name
path_pfam_counts = "../dataset/splits/%s/pfam_counts.pkl"       % dataset_settings['dataset_name']
path_dataset     = "../dataset/splits/%s/split_data.parquet"    % dataset_settings['dataset_name']
path_split       = "../dataset/splits/%s/split.json"            % dataset_settings['dataset_name']



# Load the training data
data =  pd.read_parquet(path_dataset, engine='fastparquet')

with open(path_split) as json_file:
    data_split = json.load(json_file)

# This is ugly but necessary due to wierd way we have multiple h5s
#TODO: For some reason the h5's have IDs that arent in the dataset, so these are currently filttered out. How did this happen? Probably wont have thing fixed till final presentation
#TODO: I guess its just like this now. We dont filter out all too many samples, but its still not ideal
# Load the test embeddings (once final dataset is ready, adjust this)
if dataset_settings['dataset_name'] == "easy":
    emb_path1 = "../dataset/splits/easy/easy.h5"
    emb_path2 = "../dataset/splits/easy/easy_subset_additions.h5"

    keys = []
    embeddings = []
    with h5py.File(emb_path1, "r") as f:
        keys_ = list(f.keys())
        # get all embeddings

        print(len(keys_))
        for i in range(len(keys_)):
            if i % 1000 == 0:
                print(f"Loading embedding {i+1}/{len(keys_)} for key: {key}")

            d = f[keys_[i]][:]
            if isinstance(d, np.ndarray) and keys_[i] in data.index:
                embeddings.append(d.astype(np.float32))
                keys.append(keys_[i])
            # else:
            #     print(f"Data for key {key} is not a numpy array.")

    with h5py.File(emb_path2, "r") as f:
        keys_ = list(f.keys())

        # get all embeddings
        print(len(keys_))
        for i in range(len(keys_)):
            if i % 1000 == 0:
                print(f"Loading additional embedding {i+1}/{len(keys_)} for key: {key}")

            d = f[keys_[i]][:]
            if isinstance(d, np.ndarray) and keys_[i] in data.index:
                embeddings.append(d.astype(np.float32))
                keys.append(keys_[i])
            # else:
            #     print(f"Data for key {key} is not a numpy array.")

elif dataset_settings['dataset_name'] == "complicated":

    emb_path = "../dataset/splits/complicated/data.h5"

    keys = []
    embeddings = []

    with h5py.File(emb_path, "r") as f:
        keys_ = list(f.keys())
        # get all embeddings

        print(len(keys_))
        for i in range(len(keys_)):
            if i % 1000 == 0:
                print(f"Loading embedding {i+1}/{len(keys_)} for key: {key}")

            if i > 4000:
                print("Stopping for testing purposes.")
                break

            d = f[keys_[i]][:]
            if isinstance(d, np.ndarray) and keys_[i] in data.index:
                embeddings.append(d.astype(np.float32))
                keys.append(keys_[i])
            # else:
            #     print(f"Data for key {key} is not a numpy array.")

print(len(data), "samples loaded from", path_dataset)

print(len(embeddings), "embeddings loaded from", emb_path)

num_keys_in_index = data.index.isin(keys).sum()
print(f"Number of keys in the index: {num_keys_in_index}")

# raise Exception("This is a test run, please adjust the paths and settings before running the full training.")

data = data.loc[keys]
data["embedding"] = embeddings

# Pad the embeddings
def pad_embeddings(embedding, max_length=MAX_PROTEIN_LENGTH):
    if len(embedding) < max_length:
        # Pad with zeros if the embedding is shorter than max_length
        padding = np.zeros((max_length - len(embedding), embedding.shape[1]), dtype=np.float32)
        return np.vstack((embedding, padding))
    else:
        # Truncate if the embedding is longer than max_length
        return embedding[:max_length]

data["embedding"] = data["embedding"].apply(pad_embeddings)


# Generate pfam to index mapping
pfam_to_index = {pfam: idx+1 for idx, pfam in enumerate(data["pfam_tensor"].explode().unique())}

# Convert pfams to indices and pad to max length
def convert_pfams_to_indices(pfams, pfam_to_index, max_length=MAX_PROTEIN_LENGTH):
    indices = [pfam_to_index[pfam] for pfam in pfams if pfam in pfam_to_index]
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))  # Pad with zeros
    return np.array(indices[:max_length], dtype=np.int64)

data["pfams_indices"] = data["pfam_tensor"].apply(lambda x: convert_pfams_to_indices(x, pfam_to_index))


print("Number of unique pfams in the dataset:", len(pfam_to_index)+1) # +1 for padding index
print("Number of samples in the dataset:", len(data))
print("Data preprocessing complete.")

##########################################################################################################
##########################################################################################################


# Initialize Weights & Biases (WandB) for experiment tracking
tags = [
    config['model_type'],
    f"cnn_dim_{model_settings['cnn_dim']}",
    f"kernel_size_{model_settings['kernel_size']}",
    f"dilation_{model_settings['dilation']}",

    f"batch_size_{train_settings['batch_size']}",
    f"learning_rate_{train_settings['learning_rate']}",
    
    f"max_protein_length_{dataset_settings['max_protein_length']}",
    f"num_pfams_{len(pfam_to_index)}",
    f"num_samples_{len(data)}",
    f"num_pfams_in_dataset_{len(data['pfam_tensor'].explode().unique())}",
]

wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(
    project="pp1-ProtENN2",
    tags=tags,
    config=config,
    entity='elizabeth-lochert-flx'
)


##########################################################################################################
##########################################################################################################

# Format data for training
valid_train_indices = data.index.intersection(data_split["train"])
valid_val_indices = data.index.intersection(data_split["val"])

X_train = torch.tensor(
    np.stack(data.loc[valid_train_indices]["embedding"].values),
    dtype=torch.float32
)
Y_train = torch.tensor(
    np.stack(data.loc[valid_train_indices]["pfams_indices"].values),
    dtype=torch.int64
)

X_val = torch.tensor(
    np.stack(data.loc[valid_val_indices]["embedding"].values),
    dtype=torch.float32
)
Y_val = torch.tensor(
    np.stack(data.loc[valid_val_indices]["pfams_indices"].values),
    dtype=torch.int64
)


# Define the model and move it to the appropriate device
model = ProtENN2_style(cnn_dim      = model_settings['cnn_dim'],
                       kernel_size  = model_settings['kernel_size'],
                       dilation     = model_settings['dilation'],
                       in_channels  = 1024,  # 21 for one-hot input
                       num_pfams    = len(pfam_to_index)+1).to(device)

# Define the loss function and optimizer
from sklearn.utils.class_weight import compute_class_weight

y = data["pfams_indices"].explode().dropna().astype(int).values
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

loss_cel = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)  # Ignore padding index (0)
optimizer = optim.Adam(model.parameters(), lr=train_settings['learning_rate'])

loss_cel_classic = nn.CrossEntropyLoss()   

# Validation and training loop parameters
num_epochs = train_settings['epochs']
total_ticks = 20  # Number of ticks for progress bar

batch_size = train_settings['batch_size']  # Define the batch size

# Create DataLoaders for training and validation
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
val_dataset   = torch.utils.data.TensorDataset(X_val, Y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

    print(f"\nLoss: {loss.item()}\n\n")

    # Print validation loss
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_old_loss = 0.0
    with torch.no_grad():
        for x_sequence, y in val_loader:
            x_sequence = x_sequence.to(device)
            y = y.to(device)

            y_pred = model(x_sequence)
            loss = loss_cel(y_pred.view(-1, len(pfam_to_index)+1), y.view(-1))
            val_loss += loss.item()

            old_loss = loss_cel_classic(y_pred.view(-1, len(pfam_to_index)+1), y.view(-1))
            val_old_loss += old_loss.item()
    val_loss /= len(val_loader)
    val_old_loss /= len(val_loader)

    wandb.log({"val_loss": val_loss})
    print(f"Validation Loss: {val_loss}\n")

    wandb.log({"val_old_loss": val_old_loss})
    print(f"Validation Old Loss: {val_old_loss}\n")

    # --- Learning Rate Decay ---
    for param_group in optimizer.param_groups:
        param_group['lr'] *= train_settings["lr_decay"]

    # Log the current learning rate to wandb
    wandb.log({"lr": optimizer.param_groups[0]["lr"]})


# Save the model
if train_settings['model_save_name'] is not None:
    model_save_path = f"./saved_models/{train_settings['model_save_name']}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")