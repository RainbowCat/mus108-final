from pathlib import Path

DATA = Path("datasets/lakh-pianoroll")
IDS = (DATA / "ids").glob("id_lists_*/*.txt")
for id_file in IDS:
    print(id_file)

DATASET_DIR = DATA / "lpd_5_cleansed"