import numpy as np
import torch
import wandb

from model import TheAudioBotV3
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataloader import MyDataModule
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from typing import Tuple
import random
from pytorch_lightning.loggers import WandbLogger

PARAMS = {
    "model_name": "TheAudioBotV3",
    "project_name": "TeSt Of BeSt MoDel YeS",
    "seed": 11,
    "num_epochs": 4,
    "patience": 30,
    "batch_dict": {0: 8,
                   4: 16,
                   8: 24,
                   14: 32,
                   20: 48,
                   28: 64,
                   36: 96,
                   48: 128},
    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
    "limit_train_batches": 0.1,
    "learning_rate": 0.00065,
    "optimizer": "adam",
    "loss_function": "cross_entropy",
    "activation_function": "LeakyReLU",
    "dropout": 0.09
}

random.seed(PARAMS["seed"])
torch.manual_seed(PARAMS["seed"])
np.random.seed(PARAMS["seed"])


def train(data_loader) -> None:
    model = TheAudioBotV3(lr=PARAMS["learning_rate"],
                          optimizer=PARAMS["optimizer"],
                          loss_function=PARAMS["loss_function"],
                          activation_function=PARAMS["activation_function"],
                          dropout=PARAMS["dropout"])

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models/" + PARAMS["model_name"],
        monitor="val_loss",
        mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=PARAMS["patience"],
        verbose=True,
        mode="min"
    )

    wandb_logger = WandbLogger(
        project=PARAMS["project_name"], entity="audiobots", log_model="all"
    )

    for key, val in PARAMS.items():
        if key == 'batch_dict':
            wandb_logger.experiment.config[key] = [(key_, val_) for key_, val_ in val.items()]
        else:
            wandb_logger.experiment.config[key] = val

    trainer = Trainer(
        devices="auto",
        accelerator=PARAMS["accelerator"],
        max_epochs=PARAMS["num_epochs"],
        limit_train_batches=PARAMS["limit_train_batches"],
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger
    )

    trainer.fit(model, datamodule=data_loader)
    trainer.test(model, datamodule=data_loader)
    wandb.finish()


def validate(n_folds: int = 10) -> None:
    # Load data
    data = np.load("data/raw/training.npy")
    labels = np.load("data/raw/training_labels.npy")

    train_data_temp, test_data, train_labels_temp, test_labels = train_test_split(data, labels, test_size=0.1,
                                                                                  random_state=PARAMS["seed"], shuffle=True)

    kf = KFold(n_splits=n_folds)

    for i, (train_index, test_index) in enumerate(kf.split(train_data_temp)):
        train_data = train_data_temp[train_index]
        train_labels = train_labels_temp[train_index]
        val_data = train_data_temp[test_index]
        val_labels = train_labels_temp[test_index]

        data_loader = MyDataModule(
            batch_dict=PARAMS["batch_dict"],
            device=PARAMS["accelerator"],
            train_data=train_data,
            train_labels=train_labels,
            val_data=val_data,
            val_labels=val_labels,
            test_data=test_data,
            test_labels=test_labels
        )
        train(data_loader)

if __name__ == "__main__":
    validate(10)
