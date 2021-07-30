# pylint: disable=no-member, not-callable
import logging
import os
import itertools
import random
from typing import List, Iterator, TypeVar, Union, Tuple
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset

from utils.dataset.common import (
    get_headings,
    get_viewpoints,
    load_distances,
    load_json_data,
    load_nav_graphs,
    randomize_regions,
    randomize_tokens,
    save_json_data,
    load_tokens,
    tokenize,
)
from utils.dataset.features_reader import FeaturesReader

logger = logging.getLogger(__name__)


class ZeroShotDataset(Dataset):
    def __init__(
        self,
        vln_path: str,
        tokenizer: BertTokenizer,
        features_reader: FeaturesReader,
        max_instruction_length: int,
        max_path_length: int,
        max_num_boxes: int,
        default_gpu: bool,
        highlighted_language: bool,
        **kwargs,
    ):
        # load and tokenize data (with caching)
        vln_data = load_tokens(vln_path, tokenizer, max_instruction_length)
        self._vln_by_path_id = {item["path_id"]: item for item in vln_data}
        self._instr_ids = [
            (item["path_id"], i)
            for item in vln_data
            for i in range(len(item["instructions"]))
        ]
        # load navigation graphs
        scan_list = list(set([item["scan"] for item in vln_data]))
        self._graphs = load_nav_graphs(scan_list)

        self._features_reader = features_reader
        self._max_instruction_length = max_instruction_length
        self._max_path_length = max_path_length
        self._max_num_boxes = max_num_boxes
        self._highlighted_language = highlighted_language
        self._default_gpu = default_gpu

    def __len__(self):
        return len(self._instr_ids)

    def __getitem__(self, index: int):
        path_id, stc_id = self._instr_ids[index]
        vln_item = self._vln_by_path_id[path_id]
        # get vln info
        scan_id = vln_item["scan"]
        heading = vln_item["heading"]

        # get the instruction data
        instr_tokens = torch.tensor(vln_item["instruction_tokens"][stc_id])
        instr_mask = instr_tokens > 0
        segment_ids = torch.zeros_like(instr_tokens)

        # applying a token level loss
        if self._highlighted_language:
            instr_highlights = torch.tensor(vln_item["instruction_highlights"][stc_id])
        else:
            instr_highlights = torch.tensor([])

        # get all of the paths
        selected_paths = vln_item["beams"]

        # get path features
        features, boxes, probs, masks = [], [], [], []
        for path in selected_paths:
            f, b, p, m = self._get_path_features(scan_id, path, heading)
            features.append(f)
            boxes.append(b)
            probs.append(p)
            masks.append(m)

        # convert data into tensors
        image_features = torch.tensor(features).float()
        image_boxes = torch.tensor(boxes).float()
        image_probs = torch.tensor(probs).float()
        image_masks = torch.tensor(masks).long()
        instr_tokens = instr_tokens.repeat(len(features), 1).long()
        instr_mask = instr_mask.repeat(len(features), 1).long()
        segment_ids = segment_ids.repeat(len(features), 1).long()
        instr_highlights = instr_highlights.repeat(len(features), 1).long()

        # construct null return items
        image_targets = torch.ones_like(image_probs) / image_probs.shape[-1]
        image_targets_mask = torch.zeros_like(image_masks)
        instr_targets = torch.ones_like(instr_tokens) * -1
        co_attention_mask = torch.zeros(
            2, self._max_path_length * self._max_num_boxes, self._max_instruction_length
        ).long()
        instr_id = torch.tensor([vln_item['path_id'], stc_id]).long()

        return (
            torch.tensor([]).long(),
            image_features,
            image_boxes,
            image_masks,
            image_targets,
            image_targets_mask,
            instr_tokens,
            instr_mask,
            instr_targets,
            instr_highlights,
            segment_ids,
            co_attention_mask,
            instr_id,
            torch.ones(image_features.shape[0]).bool(),
        )

    # TODO move to utils
    def _get_path_features(self, scan_id: str, path: List[str], first_heading: float):
        """ Get features for a given path. """
        headings = get_headings(self._graphs[scan_id], path, first_heading)
        # for next headings duplicate the last
        next_headings = headings[1:] + [headings[-1]]

        path_length = min(len(path), self._max_path_length)
        path_features, path_boxes, path_probs, path_masks = [], [], [], []
        for path_idx, path_id in enumerate(path[:path_length]):
            key = scan_id + "-" + path_id

            # get image features
            features, boxes, probs = self._features_reader[
                key, headings[path_idx], next_headings[path_idx],
            ]
            num_boxes = min(len(boxes), self._max_num_boxes)

            # pad features and boxes (if needed)
            pad_features = np.zeros((self._max_num_boxes, 2048))
            pad_features[:num_boxes] = features[:num_boxes]

            pad_boxes = np.zeros((self._max_num_boxes, 12))
            pad_boxes[:num_boxes, :11] = boxes[:num_boxes, :11]
            pad_boxes[:, 11] = np.ones(self._max_num_boxes) * path_idx

            pad_probs = np.zeros((self._max_num_boxes, 1601))
            pad_probs[:num_boxes] = probs[:num_boxes]

            box_pad_length = self._max_num_boxes - num_boxes
            pad_masks = [1] * num_boxes + [0] * box_pad_length

            path_features.append(pad_features)
            path_boxes.append(pad_boxes)
            path_probs.append(pad_probs)
            path_masks.append(pad_masks)

        # pad path lists (if needed)
        for path_idx in range(path_length, self._max_path_length):
            pad_features = np.zeros((self._max_num_boxes, 2048))
            pad_boxes = np.zeros((self._max_num_boxes, 12))
            pad_boxes[:, 11] = np.ones(self._max_num_boxes) * path_idx
            pad_probs = np.zeros((self._max_num_boxes, 1601))
            pad_masks = [0] * self._max_num_boxes

            path_features.append(pad_features)
            path_boxes.append(pad_boxes)
            path_probs.append(pad_probs)
            path_masks.append(pad_masks)

        return (
            np.vstack(path_features),
            np.vstack(path_boxes),
            np.vstack(path_probs),
            np.hstack(path_masks),
        )
