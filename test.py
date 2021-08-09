import json
import logging
from typing import List
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, BertTokenizer

from vilbert.vilbert import BertConfig

from utils.cli import get_parser
from utils.dataset.common import pad_packed
from utils.dataset.beam_dataset import BeamDataset
from utils.dataset.perturbate_dataset import PerturbateDataset
from utils.dataset import PanoFeaturesReader

from airbert import Airbert
from train import get_model_input, get_mask_options

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
    parser = get_parser(training=False)
    parser.add_argument(
        "--split",
        choices=["train", "val_seen", "val_unseen", "test"],
        required=True,
        help="Dataset split for evaluation",
    )
    args = parser.parse_args()

    # force arguments
    args.num_beams = 1
    args.batch_size = 1

    print(args)

    # create output directory
    save_folder = os.path.join(args.output_dir, f"run-{args.save_name}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # ------------ #
    # data loaders #
    # ------------ #

    # load a dataset
    # tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=True)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)
    if not isinstance(tokenizer, BertTokenizer):
        raise ValueError("fix mypy")
    features_reader = PanoFeaturesReader(args.img_feature)

    vln_data = f"data/task/{args.prefix}R2R_{args.split}.json"
    print(vln_data)

    dataset = BeamDataset(
        vln_path=vln_data,
        beam_path=f"data/beamsearch/{args.beam_prefix}beams_{args.split}.json",
        tokenizer=tokenizer,
        features_reader=features_reader,
        max_instruction_length=args.max_instruction_length,
        max_path_length=args.max_path_length,
        max_num_boxes=args.max_num_boxes,
        num_beams=args.num_beams,
        num_beams_strict=False,
        training=False,
        masked_vision=False,
        masked_language=False,
        default_gpu=True,
        highlighted_language=args.highlighted_language,
        num_negatives=0,
        shuffle_visual_features=args.shuffle_visual_features,
        ground_truth_trajectory=False,
    )

    if args.perturbation:
        dataset = PerturbateDataset(
            dataset=dataset,
            num_negatives=args.num_negatives,
            perturbate_path=f"data/task/{args.prefix}R2R_{args.split}.json",
            shortest_path=args.shortest_path,
            highlighted_perturbations=args.highlighted_perturbations,
        )

    data_loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- #
    # model #
    # ----- #

    config = BertConfig.from_json_file(args.config_file)
    config.cat_highlight = args.cat_highlight # type: ignore
    model = Airbert.from_pretrained(args.from_pretrained, config, default_gpu=True)
    model.cuda()
    logger.info(f"number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---------- #
    # evaluation #
    # ---------- #

    with torch.no_grad():
        all_scores = eval_epoch(model, data_loader, args)

    # save scores
    scores_path = os.path.join(save_folder, f"{args.prefix}_scores_{args.split}.json")
    json.dump(all_scores, open(scores_path, "w"))
    logger.info(f"saving scores: {scores_path}")

    # convert scores into results format
    all_results = convert_scores(
        all_scores=all_scores,
        beam_path=f"data/beamsearch/{args.beam_prefix}beams_{args.split}.json",
        add_exploration_path=args.split == "test",
    )

    # save results
    results_path = os.path.join(save_folder, f"{args.prefix}_results_{args.split}.json")
    json.dump(all_results, open(results_path, "w"))
    logger.info(f"saving results: {results_path}")


def eval_epoch(model, data_loader, args):
    device = next(model.parameters()).device

    model.eval()
    all_scores = []
    for batch in tqdm(data_loader):
        # load batch on gpu
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        instr_ids = get_instr_ids(batch)

        # get the model output
        output = model(*get_model_input(batch))
        opt_mask = get_mask_options(batch)
        vil_logit = pad_packed(output[0].squeeze(1), opt_mask)

        for instr_id, logit in zip(instr_ids, vil_logit.tolist()):
            all_scores.append((instr_id, logit))

    return all_scores


def convert_scores(all_scores, beam_path, add_exploration_path=False):
    beam_data = json.load(open(beam_path, "r"))
    instr_id_to_beams = {item["instr_id"]: item["ranked_paths"] for item in beam_data}
    instr_id_to_exploration_path = {}
    if add_exploration_path:
        instr_id_to_exploration_path = {
            item["instr_id"]: item["exploration_path"] for item in beam_data
        }

    output = []
    for instr_id, scores in all_scores:
        idx = np.argmax(scores)
        beams = instr_id_to_beams[instr_id]
        trajectory = []
        if add_exploration_path:
            trajectory += instr_id_to_exploration_path[instr_id]
        # perturbations -> we fake a wrong destination by stopping at the initial location
        if idx >= len(beams):
            trajectory = [beams[0][0]]
        else:
            trajectory += beams[idx]
        output.append({"instr_id": instr_id, "trajectory": trajectory})

    return output


# ------------- #
# batch parsing #
# ------------- #


def get_instr_ids(batch) -> List[str]:
    instr_ids = batch[12]
    return [str(item[0].item()) + "_" + str(item[1].item()) for item in instr_ids]


if __name__ == "__main__":
    main()
