"""
Build datasets for few-shot learning
"""
from itertools import groupby, combinations
import typing as t
from pathlib import Path
import shutil
from operator import itemgetter
import random
import json
import tap
import numpy as np


Split = t.Literal['train', 'val_seen', 'val_unseen']


def load_json(filename: t.Union[str, Path]):
    with open(filename, "r") as fid:
        return json.load(fid)


def save_json(data, filename: t.Union[str, Path]):
    with open(filename, "w") as fid:
        json.dump(data, fid, indent=2)




def load_data_by_scan(filename: t.Union[str, Path]):
    # Build the training set
    data = load_json(filename)
    data = sorted(data, key=itemgetter("scan"))
    data_by_scan = {
        scan: list(gen_items)
        for scan, gen_items in groupby(data, key=itemgetter("scan"))
    }
    return data_by_scan


def _create_dataset_unseen(prefix: str, data_dir: Path, beams: bool):
    # Val Unseen is kept the same
    shutil.copy(
        data_dir / "task" / "R2R_val_unseen.json", data_dir / "task" / f"{prefix}R2R_val_unseen.json"
    )

    if beams:
        shutil.copy(
            data_dir / "beamsearch" / "random+beams_val_unseen.json",
            data_dir / "beamsearch" / f"{prefix}beams_val_unseen.json",
        )

def _create_dataset_seen(scans: t.Iterable[str], prefix: str, split: Split, data_dir: Path, beams: bool):
    # Build the JSON files
    data_by_scan = load_data_by_scan(data_dir / "task" / f"R2R_{split}.json")
    fs_data = []
    fs_path_ids = set([])
    for scan in scans:
        fs_data += data_by_scan[scan]
        fs_path_ids |= {item["path_id"] for item in fs_data}
    print(prefix, split, "contains", len(fs_data), "paths")
    save_json(fs_data, data_dir / "task" / f"{prefix}R2R_{split}.json")
    
    # Extract the beams
    if beams:
        random_beams = load_json(data_dir / "beamsearch" / f"random+beams_{split}.json")
        fs_beams = []
        for beam in random_beams:
            path_id = int(beam["instr_id"].split("_")[0])
            if path_id in fs_path_ids:
                fs_beams.append(beam)
        print(prefix, split, "contains", len(fs_beams), "paths")
        save_json(fs_beams, data_dir / "beamsearch" / f"{prefix}beams_{split}.json")


def create_dataset(scans: t.Iterable[str], prefix: str, split: Split, data_dir: Path, beams: bool):
    if split in ("train", "val_seen"):
        _create_dataset_seen(scans, prefix, split, data_dir, beams)
    else:
        _create_dataset_unseen(prefix, data_dir, beams)


class Arguments(tap.Tap):
    data_dir: Path = Path("data")
    # This directory contains the following structure:
    # task/R2R_{split}.json
    # beamsearch/beamsearch/random+beams_{split}.json

    beams: bool = False


if __name__ == "__main__":
    args  = Arguments.parse_args()
    print(args)

    random.seed(1)

    # Load R2R trainset as a dictionary indexed with scan ids
    trainset = args.data_dir / 'task' / 'R2R_train.json'
    data_by_scan = load_data_by_scan(trainset)
    print("Found", len(data_by_scan),"potentiels scans")

    # We reject the buildings that have not enough instructions
    # Otherwise it is quite unfair to do FSL there
    lengths = np.array([len(data) for data in data_by_scan.values()])
    print("# samples per scans: Median", np.median(lengths), "Average", lengths.mean())

    scans = [scan for scan, data in data_by_scan.items() if len(data) > 80]
    print(f"Keeping {len(scans)} scans")

    # create 5 datasets with 1 single building on it
    for i, scan in enumerate(random.sample(scans, 5)):
        prefix = f"fsl1_{i}+"
        print(prefix, scan)

        for split in ('train', 'val_seen', 'val_unseen'):
            create_dataset([scan], prefix, split, args.data, args.beams)

    # create 5 datasets with 6 buildings on it
    seq = list(combinations(scans, 6))
    for i, subset in enumerate(random.sample(seq, 5)):
        prefix = f"fsl6_{i}+"
        print(prefix, subset)

        for split in ('train', 'val_seen', 'val_unseen'):
            create_dataset(subset, prefix, split, args.data, args.beams)
