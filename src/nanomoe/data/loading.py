import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nanomoe.config import LoaderConfig

class ShakespearDs(Dataset):
    def __init__(self, path, block_size):
        super().__init__()
        self.path = path
        self.block_size = block_size
        self.data = np.memmap(path, dtype=np.uint16, mode='r') # max token value 50256 -> uint16

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return torch.from_numpy(x).to(torch.long), torch.from_numpy(y).to(torch.long)

def get_loader_from_config(ds, loader_config: LoaderConfig):
    if loader_config.num_workers > 0:
        return DataLoader(
            ds,
            batch_size = loader_config.batch_size,
            shuffle=loader_config.shuffle,
            pin_memory=loader_config.pin_memory, # Not ideal for mac -> useful in prod env
            num_workers=loader_config.num_workers, # Not ideal for mac -> useful in prod env
            prefetch_factor=loader_config.prefetch_factor
        )
    else :
        return DataLoader(
            ds,
            batch_size = loader_config.batch_size,
            shuffle=loader_config.shuffle,
            pin_memory=loader_config.pin_memory, # Not ideal for mac -> useful in prod env
            num_workers=loader_config.num_workers, # Not ideal for mac -> useful in prod env
        )
