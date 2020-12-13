import csv
import datetime
import logging
import os
import pickle
import sys

import config
import h5py
import librosa
import numpy as np


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split("/")[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def traverse_folder(fd):
    paths = []
    names = []

    for root, dirs, files in os.walk(fd):
        for name in files:
            filepath = os.path.join(root, name)
            names.append(name)
            paths.append(filepath)

    return names, paths


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, f"{i1:04d}.log")):
        i1 += 1

    log_path = os.path.join(log_dir, f"{i1:04d}.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=log_path,
        filemode=filemode,
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    return logging


def get_metadata(audio_names, audio_paths):
    meta_dict = {
        "audio_name": audio_names,
        "audio_path": audio_paths,
        "fold": np.arange(len(audio_names)),
    }

    return meta_dict


def float32_to_int16(x):
    # assert np.max(np.abs(x)) <= 1.
    if np.max(np.abs(x)) > 1.0:
        x /= np.max(np.abs(x))
    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


class StatisticsContainer:
    def __init__(self, statistics_path):
        """Contain statistics of different training iterations."""
        self.statistics_path = statistics_path

        self.backup_statistics_path = "{}_{}.pkl".format(
            os.path.splitext(self.statistics_path)[0],
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )

        self.statistics_dict = {"validate": []}

    def append(self, iteration, statistics, data_type):
        statistics["iteration"] = iteration
        self.statistics_dict[data_type].append(statistics)

    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, "wb"))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, "wb"))
        logging.info(f"    Dump statistics to {self.statistics_path}")
        logging.info(f"    Dump statistics to {self.backup_statistics_path}")

    def load_state_dict(self, resume_iteration):
        self.statistics_dict = pickle.load(open(self.statistics_path, "rb"))

        resume_statistics_dict = {"validate": []}

        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics["iteration"] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)

        self.statistics_dict = resume_statistics_dict


class Mixup:
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator."""
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1.0 - lam)

        return np.array(mixup_lambdas)
