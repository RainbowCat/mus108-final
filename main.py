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
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence


import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch import tensor, distributions, nn, Tensor
from torch.nn import Linear, ReLU, Sequential, Softmax
from torch.utils.data import DataLoader, Dataset, TensorDataset

from panns_inference import AudioTagging, SoundEventDetection, labels
import librosa

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

