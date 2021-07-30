# pylint: disable=no-member, not-callable
import logging
import os
from pathlib import Path
from operator import itemgetter
import itertools
from itertools import groupby
import random
import copy
from typing import List, Iterator, TypeVar, Union, Tuple, Optional, Dict
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset

from utils.dataset.common import (
    get_viewpoints,
    load_distances,
    load_json_data,
    load_nav_graphs,
    save_json_data,
    load_tokens,
    tokenize,
)
from utils.dataset.features_reader import FeaturesReader
from utils.dataset.beam_dataset import BeamDataset

logger = logging.getLogger(__name__)


T = TypeVar("T")


def shuffle_different(seq: List[T]) -> Iterator[List[T]]:
    sequences = list(itertools.permutations(seq, len(seq)))
    random.shuffle(sequences)
    for s in sequences:
        l = list(s)
        if l != seq:
            yield l


def shuffle_non_adjacent(seq: List[T]) -> Iterator[List[T]]:
    n = len(seq)
    starting = {i: [j for j in range(n) if abs(j - i) > 1] for i in range(n)}
    keys = list(starting.keys())
    done = []
    while keys != []:
        idx_keys, start = random.choice(list(enumerate(keys)))
        idx_list, permute = random.choice(list(enumerate(starting[start])))

        del starting[start][idx_list]
        if starting[start] == []:
            del keys[idx_keys]

        if {start, permute} in done:
            continue
        done.append({start, permute})

        shuffled = copy.deepcopy(seq)
        shuffled[start], shuffled[permute] = shuffled[permute], shuffled[start]

        yield shuffled


def _load_noun_phrases(
    skeleton_path: Union[Path, str], tokenizer, max_instruction_length: int
) -> Optional[Dict[str, List[List[int]]]]:
    if skeleton_path == "":
        return None
    skeletons = load_tokens(skeleton_path, tokenizer, max_instruction_length)
    instr_id_to_np = {
        skeleton["instr_id"]: [
            np
            for np, is_np in zip(skeleton["instruction_tokens"], skeleton["np"])
            if is_np
        ]
        for skeleton in skeletons
    }
    return instr_id_to_np


class SpeakDataset(BeamDataset):
    def __init__(
        self,
        vln_path: str,
        tokenizer: BertTokenizer,
        features_reader: FeaturesReader,
        max_instruction_length: int,
        max_path_length: int,
        max_num_boxes: int,
        default_gpu: bool,
        skeleton_path: Union[Path, str] = "",
        shuffler: str = "different",
        separators: Tuple[str, ...] = tuple(["[SEP]", ",", "."]),
        num_splits: int = 0,
        split_id: int = 0,
    ):
        self._tokenizer = tokenizer
        self._cls, self._pad, self._sep = self._tokenizer.convert_tokens_to_ids(["[CLS]", "[PAD]", "[SEP]"])  # type: ignore
        if separators:
            self._separators: List[int] = self._tokenizer.convert_tokens_to_ids(separators)  # type: ignore
        else:
            self._separators = [self._sep]

        # load and tokenize data (with caching)
        tokenized_path = f"_tokenized_{max_instruction_length}".join(
            os.path.splitext(vln_path)
        )
        if os.path.exists(tokenized_path):
            self._vln_data = load_json_data(tokenized_path)
        else:
            self._vln_data = load_json_data(vln_path)
            tokenize(self._vln_data, tokenizer, max_instruction_length)
            save_json_data(self._vln_data, tokenized_path)

        # load navigation graphs
        scan_list = list(set([item["scan"] for item in self._vln_data]))
        self._graphs = load_nav_graphs(scan_list)
        self._distances = load_distances(scan_list)
        self._viewpoints = get_viewpoints(scan_list, self._graphs, features_reader)

        self._instr_id_to_vln = {}
        for item in self._vln_data:
            for i in range(len(item["instructions"])):
                self._instr_id_to_vln[f"{item['path_id']}_{i}"] = item
        self._instr_ids = list(self._instr_id_to_vln.keys())
        self._instr_ids = self._instr_ids[split_id:: max(1, num_splits)]

        if shuffler == "different":
            self._shuffler = shuffle_different
        elif shuffler == "nonadj":
            self._shuffler = shuffle_non_adjacent
        else:
            raise ValueError(f"Unexpected shuffling mode ({shuffler})")

        self._features_reader = features_reader
        self._instr_id_to_np = _load_noun_phrases(
            skeleton_path, tokenizer, max_instruction_length
        )
        self._max_instruction_length = max_instruction_length
        self._max_path_length = max_path_length
        self._max_num_boxes = max_num_boxes
        self._default_gpu = default_gpu

    def __len__(self):
        return len(self._instr_ids)

    def _remove_special_tokens(self, tokens: List[int]) -> List[int]:
        end = tokens.index(self._pad) - 1 if self._pad in tokens else len(tokens)
        while tokens[end - 1] in self._separators:
            end -= 1
            if end < 0:
                raise ValueError(f"Issue with tokens {tokens}")
        return tokens[1:end]

    def __getitem__(self, instr_index: int):
        instr_id = self._instr_ids[instr_index]
        vln_item = self._instr_id_to_vln[instr_id]
        path_id, instruction_index = map(int, instr_id.split("_"))

        # get vln info
        scan_id = vln_item["scan"]
        heading = vln_item["heading"]
        gt_path = vln_item["path"]

        # get the instruction data
        instr_tokens = torch.tensor(
            vln_item["instruction_tokens"][instruction_index]
        ).long()
        instr_mask = (instr_tokens > 0).long()
        instr_tokens = instr_tokens.long()
        target_tokens = copy.deepcopy(instr_tokens)
        target_mask = copy.deepcopy(instr_mask)
        segment_ids = torch.zeros_like(instr_tokens).long()

        # noun phrases
        noun_phrases = []
        if self._instr_id_to_np is not None:
            for np in self._instr_id_to_np[instr_id]:
                noun_phrases += self._remove_special_tokens(np) + [self._sep]
            noun_phrases = noun_phrases[:self._max_instruction_length]
        noun_phrases += [self._pad] * (self._max_instruction_length - len(noun_phrases))
        noun_phrase_tokens = torch.tensor(noun_phrases).long()
        np_segment_ids = torch.ones_like(noun_phrase_tokens).long()
        instr_tokens = torch.cat([noun_phrase_tokens, instr_tokens])
        segment_ids = torch.cat([np_segment_ids, segment_ids])
        instr_mask = (instr_tokens > 0).long()

        # get path features
        f, b, _, m = self._get_path_features(scan_id, gt_path, heading)
        image_features = torch.tensor(f).float()
        image_boxes = torch.tensor(b).float()
        image_masks = torch.tensor(m).long()

        # construct null return items
        co_attention_mask = torch.zeros(
            2, self._max_path_length * self._max_num_boxes, self._max_instruction_length
        ).long()

        sample_id = torch.tensor([path_id, instruction_index]).long()

        return {
            "image_features": image_features,
            "image_boxes": image_boxes,
            "image_masks": image_masks,
            "instr_tokens": instr_tokens,
            "instr_mask": instr_mask,
            "target_tokens": target_tokens,
            "target_mask": target_mask,
            "segment_ids": segment_ids,
            "co_attention_mask": co_attention_mask,
            "sample_id": sample_id,
        }
