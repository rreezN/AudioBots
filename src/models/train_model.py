import numpy as np
import torch
from model import TheAudioBotV3, TheAudioBotMini, TheAudioBotMiniV2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from dataloader import MyDataModule
import getpass
import wandb
import random
import yaml
from typing import Dict, Any
from pytorch_lightning.loggers import WandbLogger

PARAMS = {
    "model_name": "TheAudioBotMiniV2",
    "project_name": "Mini-model investigation",
    "seed": 11,
    "num_epochs": 150,
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
    "limit_train_batches": 1.0,
    "optimizer": "adam",
    "loss_function": "cross_entropy",
    "activation_function": "LeakyReLU"
}

random.seed(PARAMS["seed"])
torch.manual_seed(PARAMS["seed"])
np.random.seed(PARAMS["seed"])


def train(hparams: Dict[str, Any]) -> None:
    model = TheAudioBotMiniV2(lr=hparams["learning_rate"],
                          optimizer=PARAMS["optimizer"],
                          loss_function=PARAMS["loss_function"],
                          activation_function=PARAMS["activation_function"],
                          dropout=hparams["dropout"])

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
        callbacks=[checkpoint_callback, early_stopping_callback, TQDMProgressBar(refresh_rate=500)],
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger
    )

    data_loader = MyDataModule(batch_dict=PARAMS["batch_dict"], device=PARAMS["accelerator"])

    trainer.fit(model, datamodule=data_loader)
    trainer.test(model, datamodule=data_loader)


def train_sweep() -> None:
    with open('./sweep.yaml') as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep=sweep_config, project=PARAMS["project_name"], entity="audiobots")

    def sweep_run() -> None:
        with wandb.init() as run:
            hparams = {**run.config}  # **DEFAULT_PARAMS,
            train(hparams)

    wandb.agent(sweep_id=sweep_id, function=sweep_run, count=None)


if __name__ == "__main__":
    train_sweep()
    # hparams = {"learning_rate": 1e-3,
    #            "optimizer": "adam",
    #            "loss_function": "cross_entropy",
    #            "activation_function": "ReLU",
    #            "dropout": 0.3}
    # train(hparams)
