#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import os
import random
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from itertools import chain, product
from os import PathLike
from pathlib import Path
from typing import (Dict, Iterable, List, Mapping, NamedTuple, Optional,
                    Sequence)

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from panns_inference import AudioTagging, SoundEventDetection, labels
from torch import Tensor, distributions, nn, tensor
from torch.nn import Linear, ReLU, Sequential, Softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

# parameters
parser = argparse.ArgumentParser()
parser.add_argument("content_wav", type=Path, help="Content Wav File")
parser.add_argument("style_wav", type=Path, help="Style Wav File")
parser.add_argument("-cw", type=float, help="Content Weight", default=1)
parser.add_argument("-sw", type=float, help="Style Weight", default=1e6)
parser.add_argument("-n", type=int, help="Number of Epochs", default=10)
parser.add_argument("-lr", type=float, help="Learning Rate", default=1e-3)
args = parser.parse_args()

CHECKPOINT = Path("Cnn14_mAP=0.431.pth")

content_wav = args.content_wav
style_wav = args.style_wav

CONTENT_WEIGHT = args.cw
STYLE_WEIGHT = args.sw
# TODO add TOTAL_VARIATION

NUM_EPOCHS = args.n
LEARN_RATE = args.lr

from typing import (Dict, Iterable, List, Mapping, NamedTuple, Optional,
                    Sequence)

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from panns_inference import AudioTagging, SoundEventDetection, labels
from torch import Tensor, distributions, nn, tensor
from torch.nn import Linear, ReLU, Sequential, Softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset


class SaveActivations:
    def __init__(self):
        self.output = {}

    def __call__(self, module, _, output):
        self.output[module.__name__] = output


activation_hook=SaveActivations()

src_path = 'examples/R9_ZSCveAHg_7s.wav'
(audio, _) = librosa.core.load(src_path, sr=32000, mono=True)
audio = audio[None, :]  # (batch_size, segment_samples)


print('------ Sound event detection ------')
sed = SoundEventDetection(checkpoint_path=None, device='cuda')
framewise_output = sed.inference(audio)

net=SoundEventDetection(

        checkpoint_path=NotImplementedError,
        device='cuda' if torch.cuda.is_available() else 'cpu'
        )

net.register_forward_hook(activation_hook)

# TODO run model


################################################################################

# TODO select activations

# TODO compute loss and optimize `input`

# TODO write `input` to file


class DummySet(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, i):
        dummy = torch.tensor(1)
        return dummy

    def __len__(self):
        return 1

def StyleTransferModel(pl.LightningModule):
    def __init__(
        self,
        content_img,
        style_img,
        learn_rate,
        content_idx=CONTENT_IDX,
        style_idx=STYLE_IDX,
        content_weight=CONTENT_WEIGHT,
        style_weight=STYLE_WEIGHT,
    ):
        super().__init__()

        vgg = vgg19(pretrained=True)
        vgg = vgg.eval()  # do not change weights
        vgg = vgg.features  # drop dense layers
        self.base_model = vgg

        # initialize output styled image
        self.styled_img = nn.Parameter(content_img.clone())
        # self.styled_img = torch.rand_like(content_img)  # white noise image

        self.content_img = content_img
        self.style_img = style_img

        self.content_idx = content_idx
        self.style_idx = style_idx

        self.content_weight = content_weight
        self.style_weight = style_weight

        self.lr = learn_rate

    def on_pretrain_routine_start(self):
        """Called when the pretrain routine begins."""
        content_target, _, _ = self(self.content_img.unsqueeze(0))
        _, style_target, _ = self(self.style_img.unsqueeze(0))

        for i, c in enumerate(content_target):
            self.register_buffer(f"content_target_{i}", c.detach())
        for i, s in enumerate(style_target):
            self.register_buffer(f"style_target_{i}", gram_matrix(s).detach())

    def forward(self, img):
        x = img
        content_vals = []
        style_vals = []
        for i, layer in enumerate(self.base_model):
            x = layer(x)  # forward
            if i in self.content_idx:
                content_vals.append(x)
            if i in self.style_idx:
                style_vals.append(x)
        return content_vals, style_vals, x

    def training_step(self, batch, batch_nb):

        # unsqueeze for fake batch dimension required to fit network's input dimensions
        content_vals, style_vals, res = self(self.styled_img.unsqueeze(0))

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
        return torch.optim.Adam([self.styled_img.requires_grad_()], lr=self.lr)

    def on_train_epoch_end(self, outputs):
        # TODO save wav files
        pass
        # save_image(self.styled_img, f"tmp-{time.time()}.jpg")

def main():
    style_extractor = StyleTransferModel(content_wav, style_wav, LEARN_RATE)

    dummy_loader = DataLoader(DummySet(), batch_size=1)
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    # trainer.fit(style_extractor, dummy_loader)

    # styled_img = TF.resize(style_extractor.styled_img, content_img.size())
    # save_image(styled_img, f"output-{time.time()}.jpg")
    pass


if __name__ == "__main__":
    main()
