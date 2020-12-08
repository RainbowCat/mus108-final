### REFERENCES ###
# http://colinraffel.com/projects/lmd/
# https://github.com/craffel/midi-dataset/blob/master/Tutorial.ipynb

from pathlib import Path
import pypianoroll


def msd_id_to_dirs(msd_id: str):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return Path(msd_id[2]) / msd_id[3] / msd_id[4] / msd_id


def msd_id_to_pianoroll(dir: Path, msd_id: str):
    path = dir / msd_id_to_dirs(msd_id)
    npz = path.glob("*.npz")
    assert len(npz) == 1, f"Multiple pianoroll exists. {len(npz)}"
    return pypianoroll.load(npz[0])


# def load_all(ids_textfile:Path):
#     f = open(ids_textfile)
#     lines = f.readlines()
