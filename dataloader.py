### REFERENCES ###
# http://colinraffel.com/projects/lmd/
# https://github.com/craffel/midi-dataset/blob/master/Tutorial.ipynb

from pathlib import Path
import pypianoroll
from typing import Union

genres = [
    0,  # Pop / Rock
    1,  # Electronic
    2,  # R & B
    3,  # Blues
    4,  # Jazz
    5,  # Latin
    6,  # Reggae
    7,  # Country
    8,  # Folk
    9,  # International
    10,  # New Age
]


def msd_id_to_dirs(msd_id: str):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return Path(msd_id[2]) / msd_id[3] / msd_id[4] / msd_id


def msd_id_to_pianoroll(dir: Path, msd_id: str):
    path = dir / msd_id_to_dirs(msd_id)
    npz = path.glob("*.npz")

    assert (
        len(npz) == 1
    ), f"Multiple pianoroll exists. {len(npz)}"  # TODO do i need this?
    return pypianoroll.load(npz[0])


def read_msd_ids(f: Union[str, Path]):
    f = open(f)
    lines = f.readlines()
    return [line.strip() for line in lines]