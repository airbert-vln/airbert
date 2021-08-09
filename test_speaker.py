import json
import logging
from typing import List
from pathlib import Path
from itertools import product
import torch.multiprocessing as mp
from typing import Dict
import os
import sys
from utils.dataset.features_reader import FeaturesReader
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
from vilbert.vilbert import BertConfig
from utils.cli import get_parser
from utils.dataset.speak_dataset import SpeakDataset
from utils.dataset.bnb_speak_dataset import BnBSpeakDataset
from utils.dataset import PanoFeaturesReader, BnBFeaturesReader
from utils.dataset.common import load_json_data
from airbert import Airbert
from train_speaker import get_batch_size, get_instr_length, Batch

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def get_model_input(batch: Batch) -> Dict[str, torch.Tensor]:
    # remove the useless dimension
    image_features = batch["image_features"].squeeze(1)
    image_locations = batch["image_boxes"].squeeze(1)
    image_mask = batch["image_masks"].squeeze(1)
    instr_tokens = batch["instr_tokens"].squeeze(1)
    segment_ids = batch["segment_ids"].squeeze(1)
    instr_mask = batch["instr_mask"].squeeze(1)

    # transform batch shape
    co_attention_mask = batch["co_attention_mask"]
    co_attention_mask = co_attention_mask.view(
        -1, co_attention_mask.size(2), co_attention_mask.size(3)
    )

    return {
        "instr_tokens": instr_tokens,
        "image_features": image_features,
        "image_locations": image_locations,
        "token_type_ids": segment_ids,
        "attention_mask": torch.zeros_like(instr_mask.float()),
        "image_attention_mask": image_mask,
        "co_attention_mask": co_attention_mask,
    }


def eval_rollout(batch: Batch, model: nn.Module):
    """
    We predict the sentence step by step 
    """
    # get the model input and output
    instruction_length = get_instr_length(batch)
    batch_size = get_batch_size(batch)
    inputs = get_model_input(batch)
    # inputs["instr_tokens"][:, 1:] = 0

    for i in range(0, instruction_length - 1):
        inputs["attention_mask"][:, : i + 1] = 1

        output = model(**inputs)

        # B x N x V
        predictions = output[2].view(batch_size, -1, output[2].shape[-1])
        # B
        instr = predictions[:, i].argmax(1)

        # update instructions from predictions
        inputs["instr_tokens"][:, i + 1] = instr

    return inputs["instr_tokens"]


def test_speaker(args):
    # create output directory
    save_folder = Path(args.output_dir) / f"run-{args.save_name}"
    save_folder.mkdir(exist_ok=True)

    # ------------ #
    # data loaders #
    # ------------ #

    # load a dataset
    tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)
    if not isinstance(tokenizer, BertTokenizer):
        raise ValueError("fix mypy")

    if args.dataset == "r2r":
        dataset: Dataset = SpeakDataset(
            vln_path=f"data/task/{args.prefix}R2R_{args.split}.json",
            tokenizer=tokenizer,
            features_reader=PanoFeaturesReader(args.img_feature),
            max_instruction_length=args.max_instruction_length,
            max_path_length=args.max_path_length,
            max_num_boxes=args.max_num_boxes,
            default_gpu=True,
        )
    elif args.dataset == "bnb":
        separators = ("then", "and", ",", ".") if args.separators else ("[SEP]",)
        dataset = BnBSpeakDataset(
            trajectory_path=f"data/bnb_traj/{args.prefix}traj_bnb_{args.split}_{args.split_id}.json",
            tokenizer=tokenizer,
            features_reader=BnBFeaturesReader(args.bnb_feature),
            max_instruction_length=args.max_instruction_length,
            max_length=args.max_path_length,
            max_num_boxes=args.max_num_boxes,
            separators=separators,
            default_gpu=True,
        )
    else:
        raise ValueError(f"Unknown type of dataset: {args.dataset}")

    data_loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # ----- #
    # model #
    # ----- #

    config = BertConfig.from_json_file(args.config_file)
    config.cat_highlight = False  # type: ignore
    config.convert_mask = True  # type: ignore
    model = Airbert.from_pretrained(args.from_pretrained, config, default_gpu=True)
    model.cuda()
    logger.info(f"number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---------- #
    # evaluation #
    # ---------- #

    with torch.no_grad():
        generated_instructions = eval_epoch(model, data_loader, tokenizer)

    # update dataset for BnB:
    if args.dataset == "bnb":
        trajectories = load_json_data(
            f"data/bnb_traj/{args.prefix}traj_bnb_{args.split}_{args.split_id}.json"
        )
        for sample_id, instruction in generated_instructions.items():
            i, j = map(int, sample_id.split("_"))
            trajectories[i][j].update(instruction)
        generated_instructions = trajectories

    # save scores
    instr_path = (
        save_folder
        / f"{args.prefix}instr_{args.dataset}_{args.split}_{args.split_id}.json"
    )
    json.dump(generated_instructions, open(instr_path, "w"), indent=2)
    logger.info(f"saving generated instructions: {instr_path}")


def eval_epoch(model, data_loader, tokenizer):
    device = next(model.parameters()).device

    model.eval()
    generated_instructions = {}
    batch: Batch
    for batch in tqdm(data_loader):
        # load batch on gpu
        batch = {k: t.cuda(device=device, non_blocking=True) for k, t in batch.items()}
        instr_ids = get_instr_ids(batch)

        gen_instr = eval_rollout(batch, model)

        for instr_id, instr in zip(instr_ids, gen_instr.tolist()):
            end = len(instr)
            if 102 in instr:
                end = instr.index(102)
            if 0 in instr:
                end = min(end, instr.index(0))
            instr = instr[:end]
            generated_instructions[instr_id] = {
                "instruction_tokens": [instr],
                "instructions": [tokenizer.decode(instr)],
            }

    return generated_instructions


# ------------- #
# batch parsing #
# ------------- #


def get_instr_ids(batch: Batch) -> List[str]:
    instr_ids = batch["instr_id"]
    return [
        "_".join([str(item) for item in instr_id]) for instr_id in instr_ids.tolist()
    ]


if __name__ == "__main__":
    # command line parsing
    parser = get_parser(training=False, speaker=True, bnb=True)
    parser.add_argument(
        "--split_id", default=0, type=int, required=False,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val_seen", "val_unseen", "test"],
        required=True,
        help="Dataset split for evaluation",
    )
    args = parser.parse_args()

    test_speaker(args)
