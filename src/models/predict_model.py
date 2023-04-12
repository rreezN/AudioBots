import pickle

import click
import torch
from torch.utils.data import DataLoader, Dataset
from model import TheAudioBotV2
from tqdm import tqdm
import numpy as np
import sys
import os

sys.path.insert(0, 'C:/Users/kr_mo/OneDrive-DTU/DTU/Andet/Oticon/AudioBots/')
basedir = 'C:/Users/kr_mo/OneDrive-DTU/DTU/Andet/Oticon/AudioBots/'


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("test_filepath", type=click.Path(exists=True))
def evaluate(model_filepath, test_filepath):
    print("Evaluating model")

    # model = timm.create_model(
    #     params['hyperparameters']['model_name'],
    #     pretrained=params['hyperparameters']['pretrained'],
    #     in_chans=params['hyperparameters']['in_chans'],
    #     num_classes=params['hyperparameters']['num_classes'],
    # )

    # model.load_state_dict(torch.load(model_filepath))

    model = TheAudioBotV2.load_from_checkpoint(model_filepath)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    test_data = torch.unsqueeze(torch.tensor(np.load(os.path.join(basedir, test_filepath, "test.npy")), dtype=torch.float32), 1)
    test_labels = torch.tensor(np.load(os.path.join(basedir, test_filepath, "test_labels.npy"))).long()

    data = dataset(test_data, test_labels)
    dataloader = DataLoader(data, batch_size=100)

    correct, total = 0, 0
    for batch in tqdm(dataloader):
        x, y = batch

        preds = model(x.to(device))
        preds = preds.argmax(dim=-1)

        correct += (preds == y.to(device)).sum().item()
        total += y.numel()

    print(f"Test set accuracy {correct / total}")


if __name__ == "__main__":
    evaluate()
