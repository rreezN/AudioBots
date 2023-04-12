import logging
import pickle
from typing import Tuple
import wandb
import numpy as np
import hydra
import torch
from model import TheAudioBot
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
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
    # log.info("Training day and night")
    print("Training model")

    # model_hparams = cfg.model
    # train_hparams = cfg.training

    # print(cfg.training)

    # log.info("lr:", train_hparams.hyperparameters.lr)
    # log.info("batch size:", train_hparams.hyperparameters.batch_size)
    # torch.manual_seed(train_hparams.hyperparameters.seed)
    torch.manual_seed(seed)

    model = TheAudioBot()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        monitor="train_loss",
        mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="train_loss",
        patience=10,
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
        max_epochs=50,
        limit_train_batches=1.0,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        reload_dataloaders_every_n_epochs=1
        # precision="bf16"
    )
    # trainer = Trainer(
    #     devices="auto",
    #     accelerator=accelerator,
    #     max_epochs=train_hparams.hyperparameters.epochs,
    #     limit_train_batches=train_hparams.hyperparameters.limit_train_batches,
    #     log_every_n_steps=1,
    #     callbacks=[checkpoint_callback, early_stopping_callback],
    #     logger=wandb_logger,
    #     reload_dataloaders_every_n_epochs=1
    #     # precision="bf16"
    # )

    # log.info(f"device (accelerator): {accelerator}")
    log.info(f"device (accelerator): {accelerator}")

    train_data = torch.unsqueeze(torch.tensor(np.load(os.path.join(basedir, "data/processed/training.npy")), dtype=torch.float32), 1)
    train_labels = torch.tensor(np.load(os.path.join(basedir, "data/processed/training_labels.npy"))).long()

    train_data = dataset(train_data, train_labels)
    # train_data = dataset(train_image_data, train_images_labels.long())
    # train_loader = DataLoader(
    #     train_data,
    #     batch_size=train_hparams.hyperparameters.batch_size,
    #     num_workers=train_hparams.hyperparameters.num_workers,
    #     shuffle=True
    # )
    train_loader = DataLoader(
        train_data,
        batch_size=16,
        num_workers=1,
        shuffle=True
    )

    val_data = torch.unsqueeze(torch.tensor(np.load(os.path.join(basedir, "data/processed/val.npy")), dtype=torch.float32), 1)
    val_labels = torch.tensor(np.load(os.path.join(basedir, "data/processed/val_labels.npy"))).long()

    val_data = dataset(val_data, val_labels.long())
    val_loader = DataLoader(
        val_data,
        batch_size=16,
        num_workers=1
    )

    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    # torch.save(model, f"{os.getcwd()}/trained_model.pt")


if __name__ == "__main__":
    train()
