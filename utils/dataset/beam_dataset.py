# pylint: disable=no-member, not-callable
import logging
import os
import itertools
import random
import copy
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
    tokenize,
)
from utils.dataset.features_reader import FeaturesReader

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


class BeamDataset(Dataset):
    def __init__(
        self,
        vln_path: str,
        beam_path: str,
        tokenizer: BertTokenizer,
        features_reader: FeaturesReader,
        max_instruction_length: int,
        max_path_length: int,
        max_num_boxes: int,
        num_beams: int,
        num_beams_strict: bool,
        training: bool,
        masked_vision: bool,
        masked_language: bool,
        default_gpu: bool,
        num_negatives: int,
        ground_truth_trajectory: bool,
        highlighted_language: bool,
        shuffle_visual_features: bool,
        shuffler: str = "different",
        **kwargs,
    ):
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
        self._tokenizer = tokenizer

        # load navigation graphs
        scan_list = list(set([item["scan"] for item in self._vln_data]))

        self._graphs = load_nav_graphs(scan_list)
        self._distances = load_distances(scan_list)

        # get all of the viewpoints for this dataset
        self._viewpoints = get_viewpoints(scan_list, self._graphs, features_reader)

        # in training we only need 4 beams
        if training:
            self._num_beams = num_beams
            num_beams_strict = False

        # load beamsearch data
        temp_beam_data = load_json_data(beam_path)

        # filter beams based on length
        self._beam_data = []
        for idx, item in enumerate(temp_beam_data):
            if len(item["ranked_paths"]) >= num_beams:
                if num_beams_strict:
                    item["ranked_paths"] = item["ranked_paths"][:num_beams]
                self._beam_data.append(item)
            elif default_gpu:
                logger.warning(
                    f"skipping index: {idx} in beam data in from path: {beam_path}"
                )

        # get mapping from path id to vln index
        path_to_vln = {}
        for idx, vln_item in enumerate(self._vln_data):
            path_to_vln[vln_item["path_id"]] = idx

        # get mapping from beam to vln
        self._beam_to_vln = {}
        for idx, beam_item in enumerate(self._beam_data):
            path_id = int(beam_item["instr_id"].split("_")[0])
            if path_id not in path_to_vln:
                if default_gpu:
                    logger.warning(f"Skipping beam {beam_item['instr_id']}")
                continue
            self._beam_to_vln[idx] = path_to_vln[path_id]

        if shuffler == "different":
            self._shuffler = shuffle_different
        elif shuffler == "nonadj":
            self._shuffler = shuffle_non_adjacent
        else:
            raise ValueError(f"Unexpected shuffling mode ({shuffler})")

        self._features_reader = features_reader
        self._max_instruction_length = max_instruction_length
        self._max_path_length = max_path_length
        self._max_num_boxes = max_num_boxes
        self._training = training
        self._masked_vision = masked_vision
        self._masked_language = masked_language
        self._highlighted_language = highlighted_language
        self._ground_truth_trajectory = ground_truth_trajectory
        self._default_gpu = default_gpu
        self._shuffle_visual_features = shuffle_visual_features
        self._num_negatives = num_negatives

    def __len__(self):
        return len(self._beam_data)

    def __getitem__(self, beam_index: int):
        vln_index = self._beam_to_vln[beam_index]
        vln_item = self._vln_data[vln_index]

        # get beam info
        path_id, instruction_index = map(
            int, self._beam_data[beam_index]["instr_id"].split("_")
        )

        # get vln info
        scan_id = vln_item["scan"]
        heading = vln_item["heading"]
        gt_path = vln_item["path"]

        # get the instruction data
        instr_tokens = torch.tensor(vln_item["instruction_tokens"][instruction_index])
        instr_mask = instr_tokens > 0

        segment_ids = torch.zeros_like(instr_tokens)

        # applying a token level loss
        if self._highlighted_language:
            instr_highlights = torch.tensor(
                vln_item["instruction_highlights"][instruction_index]
            )
        else:
            instr_highlights = torch.tensor([])

        # get all of the paths
        beam_paths = []

        for ranked_path in self._beam_data[beam_index]["ranked_paths"]:
            beam_paths.append([p for p, _, _ in ranked_path])

        success = self._get_path_success(scan_id, gt_path, beam_paths)
        target: Union[List[int], int]

        # select one positive and three negative paths
        if self._training:
            # special case for data_aug with negative samples
            if "positive" in vln_item and not vln_item["positive"][instruction_index]:
                target = -1
                selected_paths = beam_paths[: self._num_beams]
                assert not self._ground_truth_trajectory, "Not compatible"

            # not enough positive or negative paths (this should be rare)
            if np.sum(success == 1) == 0 or np.sum(success == 0) < self._num_beams - 1:
                target = -1  # default ignore index
                if self._ground_truth_trajectory:
                    selected_paths = [self._vln_data[vln_index]["path"]] + beam_paths[
                        : self._num_beams - 1
                    ]
                else:
                    selected_paths = beam_paths[: self._num_beams]
            else:
                target = 0
                selected_paths = []
                # first select a positive
                if self._ground_truth_trajectory:
                    selected_paths.append(self._vln_data[vln_index]["path"])
                else:
                    idx = np.random.choice(np.where(success == 1)[0])  # type: ignore
                    selected_paths.append(beam_paths[idx])
                # next select three negatives
                idxs = np.random.choice(  # type: ignore
                    np.where(success == 0)[0], size=self._num_beams - 1, replace=False
                )
                for idx in idxs:
                    selected_paths.append(beam_paths[idx])

            # shuffle the visual features from the ground truth as a free negative path
            if self._shuffle_visual_features:
                path = self._vln_data[vln_index]["path"]
                # selected_paths += [
                #     corr
                #     for corr, _ in zip(shuffle_gen(path), range(self._num_negatives))
                # ]
                for corr, _ in zip(self._shuffler(path), range(self._num_negatives)):
                    selected_paths += [corr]

        else:
            target = success
            selected_paths = beam_paths

            # This should be used only for testing the influence of shuffled sequences!
            if self._shuffle_visual_features:
                if isinstance(target, int):
                    raise ValueError("fix mypy")

                # we shuffled all positive trajectories
                # the target is now zero everywhere
                for i in np.arange(len(success))[success.astype("bool")]:
                    if i > self._num_negatives:
                        break
                    selected_paths.append(next(self._shuffler(selected_paths[i])))
                    target = np.append(target, 0)

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

        # randomly mask image features
        if self._masked_vision:
            image_features, image_targets, image_targets_mask = randomize_regions(
                image_features, image_probs, image_masks
            )
        else:
            image_targets = torch.ones_like(image_probs) / image_probs.shape[-1]
            image_targets_mask = torch.zeros_like(image_masks)

        # randomly mask instruction tokens
        if self._masked_language:
            instr_tokens, instr_targets = randomize_tokens(
                instr_tokens, instr_mask, self._tokenizer
            )
        else:
            instr_targets = torch.ones_like(instr_tokens) * -1

        # construct null return items
        co_attention_mask = torch.zeros(
            2, self._max_path_length * self._max_num_boxes, self._max_instruction_length
        ).long()
        instr_id = torch.tensor([path_id, instruction_index]).long()

        return (
            torch.tensor(target).long(),
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

    def _get_path_success(self, scan_id, path, beam_paths, success_criteria=3):
        d = self._distances[scan_id]
        success = np.zeros(len(beam_paths))
        for idx, beam_path in enumerate(beam_paths):
            if d[path[-1]][beam_path[-1]] < success_criteria:
                success[idx] = 1
        return success

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
