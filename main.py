from pathlib import Path
import dataloader
import pypianoroll

DATA = Path("datasets/lakh-pianoroll")
IDS_DIR = DATA / "id_lists_lastfm"

DATASET_DIR = DATA / "lpd_5_cleansed"

f = IDS_DIR / "id_list_00s.txt"
chosen_ids = f.open().readlines()
print(f"{len(chosen_ids)} are chosen.")

# dataset = {}
# for f in ids:
#     f = IDS_DIR / "id_list_00s.txt"
#     chosen_ids = f.open().readlines()
#     for i in chosen_ids:
#         try:
#             npz_dir = dataloader.msd_id_to_dirs(i)
#             npz = dataloader.msd_id_to_pianoroll(npz_dir)
#             dataset[i] = npz
#         except:
#             continue

# print(f"{len(dataset)}")
