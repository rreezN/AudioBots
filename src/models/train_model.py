import logging
from typing import Tuple
import numpy as np
import torch
from model import TheAudioBotV3
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from dataloader import MyDataModule
import sys
import os

sys.path.insert(0, 'C:/Users/kr_mo/OneDrive-DTU/DTU/Andet/Oticon/AudioBots/')
basedir = 'C:/Users/kr_mo/OneDrive-DTU/DTU/Andet/Oticon/AudioBots/'

seed = 11
# wandb.login(key='5b7c4dfaaa3458ff59ee371774798a737933dfa9')


class dataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.data = images
        self.labels = labels

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[item].float(), self.labels[item]

    def __len__(self) -> int:
        return len(self.data)


# @hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def train() -> None:
    logging.info("Training model")

    torch.manual_seed(seed)

    model = TheAudioBotV3()
    model_name = "TheAudioBotV3"

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models/" + model_name,
        monitor="val_loss",
        mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=50,
        verbose=True,
        mode="min"
    )
    # accelerator = "gpu" if train_hparams.hyperparameters.cuda else "cpu"
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # wandb_logger = WandbLogger(
    #     project="Final-Project", entity="dtu-mlopsproject", log_model="all"
    # )
    # for key, val in train_hparams.hyperparameters.items():
    #     wandb_logger.experiment.config[key] = val
    trainer = Trainer(
        devices="auto",
        accelerator=accelerator,
        max_epochs=200,
        # limit_train_batches=1.0,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        reload_dataloaders_every_n_epochs=1
    )

    logging.info(f"device (accelerator): {accelerator}")
    data_loader = MyDataModule(batch_dict={0: 8,
                                           4: 16,
                                           8: 24,
                                           14: 32,
                                           20: 48,
                                           28: 64,
                                           36: 96,
                                           48: 128})

    trainer.fit(model, datamodule=data_loader)
    logging.info("Training complete")


if __name__ == "__main__":
    train()
