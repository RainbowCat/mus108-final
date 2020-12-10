from __future__ import annotations

import argparse
import itertools
import os
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

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from numpy.compat.py3k import asstr
from PIL import Image
from torch import Tensor, distributions, nn, tensor
from torch.nn import Linear, ReLU, Sequential, Softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

lakh = Path("datasets/LakhNES")
script_dir = lakh / "scripts"

model_path = lakh / "model.pt"
code_model_dir = os.path.abspath(os.path.join(script_dir, "model"))
code_utils_dir = os.path.join(code_model_dir, "utils")
sys.path.extend([code_model_dir, code_utils_dir])
model = torch.load("datasets/LakhNES/model.pt")