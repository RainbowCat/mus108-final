#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from itertools import chain, product
from os import PathLike
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence

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

# TODO add TOTAL_VARIATION

TMP_DIR = Path("genre_activations")
TMP_DIR.mkdir(exist_ok=True)
# OUT_DIR = Path("output")
# OUT_DIR.mkdir(exist_ok=True)


class SaveActivations:
    def __init__(self):
        self.output = []

    def __call__(self, module, _, output) -> None:
        self.output.append(output)

    def clear(self) -> None:
        self.output.clear()


def inference(self, audio):
    if "float" in str(audio.dtype):
        audio = torch.Tensor(audio)
    elif "int" in str(audio.dtype):
        audio = torch.LongTensor(audio)

    audio = audio.to(self.device)

    output_dict = self.model(audio, None)
    print(output_dict)

    framewise_output = output_dict["framewise_output"].data.cpu().numpy()

    return framewise_output


SoundEventDetection.__call__ = inference


class StyleTransferModel(pl.LightningModule):
    def __init__(
        self, content_wav, style_wav, content_weight, style_weight, learn_rate
    ):
        super().__init__()
        self.content_wav = content_wav
        self.style_wav = style_wav

        self.save_hyperparameters()

        self.register_buffer(
            "content",
            torch.tensor(librosa.core.load(content_wav, sr=32000, mono=True)[0]),
        )
        self.register_buffer(
            "style",
            torch.tensor(librosa.core.load(style_wav, sr=32000, mono=True)[0]),
        )
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.lr = learn_rate

        self.base_model = SoundEventDetection(
            checkpoint_path=None, device=self.device
        ).model
        self.base_model = self.base_model.eval()  # do not change weights

        self.activation_hook = SaveActivations()

        def apply_hook(layer):
            if isinstance(layer, Conv2d):
                layer.register_forward_hook(self.activation_hook)

        self.base_model.apply(apply_hook)

        # initialize output styled audio
        self.styled = nn.Parameter(self.content.clone())

    def on_pretrain_routine_start(self) -> None:
        """Called when the pretrain routine begins."""
        self(self.content.unsqueeze(0))
        for i, c in enumerate(self.activation_hook.output):
            self.register_buffer(f"content_target_{i}", c.detach())

        with open(
            TMP_DIR / f"{self.content_wav.stem}-content_activations.pkl", "wb"
        ) as f:
            pickle.dump(self.activation_hook.output, f)
        self.activation_hook.clear()

        self(self.style.unsqueeze(0))
        for i, s in enumerate(self.activation_hook.output):
            self.register_buffer(f"style_target_{i}", gram_matrix(s).detach())
        with open(TMP_DIR / f"{self.style_wav.stem}-style_activations.pkl", "wb") as f:
            pickle.dump(self.activation_hook.output, f)
        self.activation_hook.clear()

    def forward(self, x):
        return self.base_model(x)

    def training_step(self, batch, batch_nb):
        # XXX must be here to avoid accumulating state after end
        self.activation_hook.clear()

        # unsqueeze for fake batch dimension required to fit network's input dimensions
        self(self.styled.unsqueeze(0))
        activation_vals = self.activation_hook.output

        # TODO see shape and pick
        content_vals = activation_vals
        style_vals = activation_vals

        content_loss = sum(
            F.mse_loss(c, getattr(self, f"content_target_{i}"))
            for i, c in enumerate(content_vals)
        )
        style_loss = sum(
            F.mse_loss(gram_matrix(s), getattr(self, f"style_target_{i}"))
            for i, s in enumerate(style_vals)
        )
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss

        self.log("content_loss", content_loss, prog_bar=True)
        self.log("style_loss", style_loss, prog_bar=True)
        self.log("total_loss", total_loss, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam([self.styled.requires_grad_()], lr=self.lr)

    def on_train_epoch_end(self, outputs):
        # XXX DO NOTHING

        # assert self.activation_hook.output, self.activation_hook_output

        # stem = f"{time.time()}"

        # if random.random() < 0.3:
        #     with open(TMP_DIR / "{stem}.pkl", "wb") as f:
        #         pickle.dump(self.activation_hook.output, f)

        #     soundfile.write(  # save wav files
        #         file=TMP_DIR / f"{stem}.wav",
        #         data=self.styled.data.detach().cpu().numpy(),
        #         samplerate=32_000,
        #     )
        pass


def gram_matrix(input):
    """REFERENCE: <https://pytorch.org/tutorials/advanced/neural_style_tutorial.html/>"""
    b, c, h, w = input.shape

    features = input.view(b * c, h * w)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    return G / input.numel()


class DummySet(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, i):
        dummy = torch.tensor(1)
        return dummy

    def __len__(self):
        return 1


def main(args) -> None:
    style_extractor = StyleTransferModel(
        args.content_wav,
        args.style_wav,
        args.content_weight,
        args.style_weight,
        args.learn_rate,
    )

    dummy_loader = DataLoader(DummySet(), batch_size=1)
    trainer = pl.Trainer(max_epochs=args.num_epochs, gpus=torch.cuda.device_count())
    trainer.fit(style_extractor, dummy_loader)

    styled = style_extractor.styled.data.detach().cpu().numpy()
    soundfile.write(  # save wav files
        file=f"{args.content_wav.stem}-{args.style_wav.stem},sw={args.style_weight},ne={args.num_epochs},lr={args.learn_rate}.wav",
        data=styled,
        samplerate=32_000,
    )


if __name__ == "__main__":
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "content_wav", nargs="?", type=Path, default=Path("data/SpringDay_30.wav")
    )
    parser.add_argument(
        "style_wav", nargs="?", type=Path, default=Path("data/HighwayToHell_30.wav")
    )
    parser.add_argument(
        "content_weight", nargs="?", type=float, help="Content Weight", default=1e0
    )
    parser.add_argument(
        "style_weight", nargs="?", type=float, help="Style Weight", default=1e6
    )
    parser.add_argument(
        "num_epochs", nargs="?", type=int, help="Number of Epochs", default=10
    )
    parser.add_argument(
        "learn_rate", nargs="?", type=float, help="Learning Rate", default=1e-1
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("Cnn14_mAP=0.431.pth"))
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    args = parser.parse_args()
    main(args)
