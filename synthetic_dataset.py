import os

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self):

        # Load metadata from YAML file
        with open("metadata.yaml", "r") as f:
            self.meta = yaml.safe_load(f)

        self.num_shards = self.meta["num_shards"]
        self.num_samples = self.meta["num_samples"]
        self.num_features = self.meta["num_features"]
        self.output_dir = self.meta["output_dir"]
        self.total_samples = self.num_shards * self.num_samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        shard_idx = idx // self.num_samples
        sample_idx = idx % self.num_samples

        filename = os.path.join(self.output_dir, f"shard_{shard_idx:02d}.npy")
        
        data = np.load(filename)
        sample = data[sample_idx]
        return torch.from_numpy(sample)