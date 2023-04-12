from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
import numpy as np
import os
import sys

sys.path.insert(0, '//')
basedir = 'C:/Users/kr_mo/OneDrive-DTU/DTU/Andet/Oticon/AudioBots/'


class dataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.data = images
        self.labels = labels

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[item].float(), self.labels[item]

    def __len__(self) -> int:
        return len(self.data)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_dict):
        super().__init__()
        self.batch_dict = batch_dict
        self.current_batch_size = batch_dict[0]

    def setup(self, stage=None):
        train_data = torch.unsqueeze(torch.tensor(np.load(os.path.join(basedir, "data/processed/training.npy")), dtype=torch.float32), 1)
        train_labels = torch.tensor(np.load(os.path.join(basedir, "data/processed/training_labels.npy"))).long()
        self.train_data = dataset(train_data, train_labels)

        val_data = torch.unsqueeze(
            torch.tensor(np.load(os.path.join(basedir, "data/processed/val.npy")), dtype=torch.float32), 1)
        val_labels = torch.tensor(np.load(os.path.join(basedir, "data/processed/val_labels.npy"))).long()
        self.val_data = dataset(val_data, val_labels.long())

    def train_dataloader(self):
        # Update current_batch_size based on epoch number
        arr = np.array(list(self.batch_dict.keys()))
        key = np.max(arr[arr <= self.trainer.current_epoch])
        self.current_batch_size = self.batch_dict[key]
        # Return DataLoader with current_batch_size
        return DataLoader(self.train_data, batch_size=self.current_batch_size, num_workers=1, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.current_batch_size, num_workers=1, shuffle=True)