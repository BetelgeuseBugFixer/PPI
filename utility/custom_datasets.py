import pandas as pd
from torch.utils.data import Dataset
import h5py
import torch
from torch.utils.data import Dataset

class OneHotDataset(Dataset):
    def __init__(self, file_dir, transform=None, target_transform=None):
        self.data = pd.read_csv(file_dir, names=["protein","label","seq"])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data["seq"][idx]
        label = self.data["label"][idx]
        if self.transform:
            seq = self.transform(seq)
        if self.target_transform:
            label = self.target_transform(label)
        return seq, label
    

class ProtT5Dataset(Dataset):
    def __init__(self, embeddings_h5_file_path, labels):
        self.embeddings = h5py.File(embeddings_h5_file_path, 'r')
        self.labels = labels
        self.keys = list(self.embeddings.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        embedding = self.embeddings[key][:]
        label = self.labels[idx]
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def close(self):
        self.embeddings.close()