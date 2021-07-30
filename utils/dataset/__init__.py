from pathlib import Path
from typing import Union
from .pano_features_reader import PanoFeaturesReader
from .features_reader import FeaturesReader
from .bnb_features_reader import BnBFeaturesReader


def load_features_reader(
    src: str, path: Union[Path, str], in_memory: bool
) -> FeaturesReader:
    if src == "r2r":
        return PanoFeaturesReader(path=path, in_memory=in_memory)
    elif src == "bnb":
        return BnBFeaturesReader(path=path, in_memory=in_memory)
    else:
        raise ValueError(f"Unknown feature reader ({str})")

