"""
Test training on BnB
"""
import json
import logging
from typing import List
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from apex.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from transformers import AutoTokenizer, BertTokenizer

from vilbert.vilbert import BertConfig

from utils.cli import get_parser
from utils.dataset.common import pad_packed, save_json_data
from utils.dataset import BnBFeaturesReader
from utils.dataset.bnb_dataset import BnBDataset
from utils.dataset import PanoFeaturesReader

from airbert import VLNBert
from train import get_model_input, get_mask_options, get_target

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main():
    # ----- #
    # setup #
    # ----- #

    # command line parsing
    parser = get_parser(training=False, bnb=True)
    args = parser.parse_args()
    print(args)

    # create output directory
    save_folder = os.path.join(args.output_dir, f"run-{args.save_name}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # get device settings
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of synchronizing
        # nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
        n_gpu = 1

    # check if this is the default gpu
    default_gpu = True
    if args.local_rank != -1 and dist.get_rank() != 0:
        default_gpu = False
    if default_gpu:
        logger.info(f"Playing with {n_gpu} GPUs")

    # ------------ #
    # data loaders #
    # ------------ #

    # load a dataset
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    if not isinstance(tokenizer, BertTokenizer):
        raise ValueError("Fix mypy issue")
    features_reader = BnBFeaturesReader(args.bnb_feature)
    caption_path = f"data/bnb/{args.prefix}bnb_test.json"
    logger.info(f"Using captions from {caption_path}")

    separators = ("then", "and", ",", ".") if args.separators else ("[SEP]",)
    testset_path = f"data/bnb/{args.prefix}testset.json"

    dataset = BnBDataset(
        caption_path=caption_path,
        testset_path=testset_path,
        tokenizer=tokenizer,
        skeleton_path=args.skeleton,
        features_reader=features_reader,
        max_instruction_length=args.max_instruction_length,
        max_length=args.max_path_length,
        min_length=args.min_path_length,
        max_captioned=args.max_captioned,
        min_captioned=args.min_captioned,
        max_num_boxes=args.max_num_boxes,
        num_positives=1,
        num_negatives=args.num_negatives,
        masked_vision=False,
        masked_language=False,
        training=False,
        shuffler=args.shuffler,
        out_listing=args.out_listing,
        separators=separators,
    )

    logger.info("Loading val datasets")

    # adjust the batch size for distributed training
    batch_size = args.batch_size
    if args.local_rank != -1:
        batch_size = batch_size // dist.get_world_size()
    if default_gpu:
        logger.info(f"batch_size: {batch_size}")

    if args.local_rank == -1:
        sampler = SequentialSampler(dataset)
    else:
        sampler = DistributedSampler(dataset)

    data_loader = DataLoader(
        dataset,
        shuffle=False,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- #
    # model #
    # ----- #

    config = BertConfig.from_json_file(args.config_file)
    config.cat_highlight = args.cat_highlight  # type: ignore
    model = VLNBert.from_pretrained(args.from_pretrained, config, default_gpu=True)
    logger.info(f"number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.to(device)
    if args.local_rank != -1:
        model = DDP(model, delay_allreduce=True)
        if default_gpu:
            logger.info("using distributed data parallel")
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        if default_gpu:
            logger.info("using data parallel")

    # ---------- #
    # evaluation #
    # ---------- #

    with torch.no_grad():
        all_scores = eval_epoch(model, data_loader)

    # save scores
    scores_path = os.path.join(save_folder, f"{args.prefix}_scores.json")
    save_json_data(all_scores, scores_path)
    logger.info(f"saving scores: {scores_path}")


def eval_epoch(model, data_loader):
    device = next(model.parameters()).device

    model.eval()
    all_scores = {}
    counter = 0
    for batch in tqdm(data_loader):
        # load batch on gpu
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        listing_ids = get_listing_ids(batch)

        # get the model output
        output = model(*get_model_input(batch))
        opt_mask = get_mask_options(batch)
        instr_tokens = get_instr_tokens(batch)
        target = get_target(batch)
        vil_logit = pad_packed(output[0].squeeze(1), opt_mask)

        all_scores[listing_ids[0]] = {
            "logit": vil_logit[0].tolist(),
            "target": target[0].tolist(),
            "instr": instr_tokens[0].tolist(),
        }

        # DEBUG
        counter += 1
        if counter == 20:
            break
    return all_scores


def get_listing_ids(batch) -> List[str]:
    instr_ids = batch[12]
    return [str(int(item)) for item in instr_ids]

def get_instr_tokens(batch):
    return batch[6]



if __name__ == "__main__":
    main()
