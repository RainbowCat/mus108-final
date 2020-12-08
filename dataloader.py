### REFERENCES ###
# http://colinraffel.com/projects/lmd/
# https://github.com/craffel/midi-dataset/blob/master/Tutorial.ipynb

# Imports
import numpy as np
import glob
import pypianoroll

# import matplotlib.pyplot as plt
# %matplotlib inline
# import pretty_midi
# import librosa
# import mir_eval
# import mir_eval.display
# import tables
# import IPython.display
import os
from pathlib import Path

# import json

# Local path constants
DATA_PATH = Path("data")
RESULTS_PATH = Path("results")
# Path to the file match_scores.json distributed with the LMD
SCORE_FILE = RESULTS_PATH / "match_scores.json"

# Utility functions for retrieving paths
def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return Path(msd_id[2]) / msd_id[3] / msd_id[4] / msd_id


# def msd_id_to_mp3(msd_id):
#     """Given an MSD ID, return the path to the corresponding mp3"""
#     return DATA_PATH / "msd" / "mp3" / (msd_id_to_dirs(msd_id) + ".mp3")


# def msd_id_to_h5(msd_id):
#     """Given an MSD ID, return the path to the corresponding h5"""
#     return RESULTS_PATH / "lmd_matched_h5" / (msd_id_to_dirs(msd_id) + ".h5")


# def get_midi_path(msd_id, midi_md5, kind):
#     """Given an MSD ID and MIDI MD5, return path to a MIDI file.
#     kind should be one of 'matched' or 'aligned'."""
#     return (
#         RESULTS_PATH
#         / "lmd_{}".format(kind)
#         / (msd_id_to_dirs(msd_id), midi_md5 + ".mid")
#     )


def msd_id_to_pianoroll(msd_id):
    path = msd_id_to_dirs(msd_id)
    npz = glob.glob("*.npz")
    assert len(npz) == 1, f"Multiple pianoroll exists. {len(npz)}"
    return pypianoroll.load(npz[0])