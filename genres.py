#!/usr/bin/env python3

from __future__ import annotations

import itertools
import os
import pickle
import random
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pytorch_lightning as pl
import soundfile
import torch
import torch.nn.functional as F
from panns_inference import AudioTagging, SoundEventDetection, labels
from torch import Tensor, distributions, nn, tensor
from torch.nn import Conv2d, Linear, ReLU, Sequential, Softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

GENRES_FOLDER = Path("genres")
TMP_DIR = Path("genre_activations")
TMP_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SaveActivations:
    def __init__(self):
        self.output = []

    def __call__(self, module, _, output) -> None:
        self.output.append(output)

    def clear(self) -> None:
        self.output.clear()


activation_hook = SaveActivations()


def apply_hook(layer):
    if isinstance(layer, Conv2d):
        layer.register_forward_hook(activation_hook)


base_model = SoundEventDetection(checkpoint_path=None, device=DEVICE).model.eval()
base_model.apply(apply_hook)

for genre in [x for x in GENRES_FOLDER.iterdir() if x.is_dir()]:
    with torch.no_grad():
        input = torch.stack(
            [
                torch.tensor(librosa.core.load(wav, sr=32000, mono=True)[0])
                for wav in genre.glob("*.wav")
            ],
            device=DEVICE,
        )

        base_model(tensor)
        with open(f"{genre.name}.pkl", "wb") as f:
            pickle.dump(f.activation_hook.output)

        activation_hook.clear()
