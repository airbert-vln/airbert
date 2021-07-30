from typing import Tuple, Union, Sequence, List
from pathlib import Path
import pickle
import lmdb
import numpy as np


class FeaturesReader:
    def __init__(self, path: Union[Path, str, Sequence[Union[Path, str]]]):
        if isinstance(path, (Path, str)):
            path = [path]

        # open database
        self.envs = [
            lmdb.open(
                str(p),
                readonly=True,
                readahead=False,
                max_readers=20,
                lock=False,
                map_size=int(1e9),
            )
            for p in path
        ]

        # get keys
        self.keys = {}
        for i, env in enumerate(self.envs):
            with env.begin(write=False, buffers=True) as txn:
                bkeys = txn.get("keys".encode())
                if bkeys is None:
                    raise RuntimeError("Please preload keys in the LMDB")
                for k in pickle.loads(bkeys):
                    self.keys[k.decode()] = i

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, keys: Tuple) -> List:
        for key in keys:
            if not isinstance(key, str) or key not in self.keys:
                raise TypeError(f"invalid key: {key}")

        env_idx = [self.keys[key] for key in keys]
        items = [None] * len(keys)

        # we minimize the number of connections to an LMDB
        for idx in set(env_idx):
            with self.envs[idx].begin(write=False) as txn:
                for i, (idx_i, key) in enumerate(zip(env_idx, keys)):
                    if idx_i != idx:
                        continue
                    item = txn.get(key.encode())
                    if item is None:
                        continue
                    items[i] = pickle.loads(item)

        return items
