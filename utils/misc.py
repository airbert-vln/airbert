import random
import logging
from pathlib import Path
import sys
import numpy as np
import torch
import torch.distributed as dist


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )
    logger = logging.getLogger(name)
    return logger



def set_seed(args) -> None:
    if args.seed:
        print("Setting the seed")
        seed = args.seed
        if args.local_rank != -1:
            seed += args.local_rank
        torch.manual_seed(seed)
        np.random.seed(seed)  # type: ignore
        random.seed(seed)

def is_default_gpu(args) -> bool:
    """
    check if this is the default gpu
    """
    return args.local_rank == -1 or dist.get_rank() != 0


def get_output_dir(args) -> Path:
    odir = Path(args.output_dir) / f"run-{args.save_name}"
    odir.mkdir(exist_ok=True, parents=True)
    return odir.resolve()

