from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
from torchvision import transforms
import numpy as np
import os

def standardiseTransform(data, mean, std):
    # Standardise data
    return (data - mean.view(1, 1, 32, 1)) / std.view(1, 1, 32, 1)

class dataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        self.data = data
        self.labels = labels

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[item].float(), self.labels[item]

    def __len__(self) -> int:
        return len(self.data)

class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_dict):
        super().__init__()
        # batch_dict contains the batch scheduling
        self.batch_dict = batch_dict
        self.current_batch_size = batch_dict[0]

    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders

        train_data = torch.unsqueeze(torch.tensor(np.load("data/processed/training.npy"), dtype=torch.float32), 1)
        train_labels = torch.tensor(np.load("data/processed/training_labels.npy")).long()
        
        # Standardisation parameters
        mu = torch.mean(train_data, axis=(0,1,3))
        std = torch.std(train_data, axis=(0,1,3))

        # Standardise train data
        train_data = standardiseTransform(train_data, mu, std)
        self.train_data = dataset(train_data, train_labels)

        # Standardise val data
        val_data = torch.unsqueeze(
            torch.tensor(np.load("data/processed/val.npy"), dtype=torch.float32), 1)
        val_labels = torch.tensor(np.load("data/processed/val_labels.npy")).long()
        val_data = standardiseTransform(val_data, mu, std)
        self.val_data = dataset(val_data, val_labels.long())

        # Standardise test data
        test_data = torch.unsqueeze(
            torch.tensor(np.load("data/processed/test.npy"), dtype=torch.float32), 1)
        test_labels = torch.tensor(np.load("data/processed/test_labels.npy")).long()
        test_data = standardiseTransform(test_data, mu, std)
        self.test_data = dataset(test_data, test_labels.long())

    def train_dataloader(self):
        # Update current_batch_size based on epoch number
        arr = np.array(list(self.batch_dict.keys()))
        key = np.max(arr[arr <= self.trainer.current_epoch])
        self.current_batch_size = self.batch_dict[key]
        # Return DataLoader with current_batch_size
        return DataLoader(self.train_data, batch_size=self.current_batch_size, num_workers=1, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.current_batch_size, num_workers=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.current_batch_size, num_workers=1, shuffle=False)
