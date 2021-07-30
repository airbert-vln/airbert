import logging
import os
import random
from typing import List, Dict, Any, Callable, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from utils.dataset.common import (
    load_tokens,
    randomize_regions,
    randomize_tokens,
)
from utils.dataset.beam_dataset import BeamDataset

logger = logging.getLogger(__name__)


class PerturbateDataset(Dataset):
    def __init__(
        self,
        dataset: BeamDataset,
        perturbate_path: str,
        shortest_path: bool,
        num_negatives: int = 2,
        highlighted_perturbations: bool = False,
    ):
        # Load perturbations and tokenize instructions (with caching)
        self._perturbate_data = load_tokens(
            perturbate_path, dataset._tokenizer, dataset._max_instruction_length
        )
        self._num_negatives = num_negatives
        self._highlighted_perturbations = highlighted_perturbations
        self._shortest_path = shortest_path
        self._dataset = dataset

        # check we have the same index for VLN data and for perturbations
        assert len(self._dataset._vln_data) == len(self._perturbate_data)

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, beam_index: int):
        # get classic batch
        (
            target,
            image_features,
            image_boxes,
            image_masks,
            image_targets,
            image_target_masks,
            instr_tokens,
            instr_masks,
            instr_targets,
            instr_highlights,
            segment_ids,
            co_attention_masks,
            instr_id,
            opt_masks,
        ) = self._dataset[beam_index]

        vln_index = self._dataset._beam_to_vln[beam_index]
        instr_idx = int(self._dataset._beam_data[beam_index]["instr_id"].split("_")[1])
        item = self._perturbate_data[vln_index]
        num_candidates = len(item["perturbation_tokens"][instr_idx])
        opt_mask = torch.zeros(self._num_negatives).bool()

        # add perturbations
        if num_candidates > 0:
            indices = list(range(num_candidates))
            if num_candidates > self._num_negatives:
                indices = random.sample(indices, self._num_negatives)
            instr_token = torch.tensor(
                [item["perturbation_tokens"][instr_idx][idx] for idx in indices]
            ).long()
            instr_mask = (instr_token > 0).long()
            opt_mask[: len(indices)] = True
            segment_id = torch.zeros_like(instr_token)
            if self._highlighted_perturbations:
                instr_highlight = torch.tensor(
                    [
                        item["perturbation_highlight_masks"][instr_idx][idx]
                        for idx in indices
                    ]
                ).long()
            else:
                instr_highlight = torch.zeros_like(instr_token).long()
        else:
            instr_token = torch.zeros_like(instr_tokens[:1])
            instr_mask = torch.zeros_like(instr_masks[:1])
            instr_highlight = torch.zeros_like(instr_highlights[:1])
            segment_id = torch.zeros_like(segment_ids[:1])

        # pad tensors in case of missing perturbations
        pad_len = self._num_negatives - instr_token.shape[0]
        instr_token = F.pad(instr_token, (0, 0, pad_len, 0))
        instr_mask = F.pad(instr_mask, (0, 0, pad_len, 0))
        segment_id = F.pad(segment_id, (0, 0, pad_len, 0))
        if self._dataset._highlighted_language:
            instr_highlight = F.pad(instr_highlight, (0, 0, pad_len, 0))
        else:
            instr_highlight = torch.zeros((self._num_negatives, 0)).long()

        # Get golden path from ground truth data, because a beam path might be correct
        # but it might not exactly follow the instructions
        image_feature, image_box, image_prob, image_mask = self._get_ground_truth_trajectory(
            beam_index
        )

        # repeat visual tensors to match with the number of perturbations
        image_feature = image_feature.repeat(self._num_negatives, 1, 1).float()
        image_box = image_box.repeat(self._num_negatives, 1, 1).float()
        image_prob = image_prob.repeat(self._num_negatives, 1, 1).float()
        image_mask = image_mask.repeat(self._num_negatives, 1).long()

        # randomly mask image features
        if self._dataset._masked_vision:
            image_feature, image_target, image_target_mask = randomize_regions(
                image_feature, image_prob, image_mask
            )
        else:
            image_target = torch.ones_like(image_prob) / image_prob.shape[-1]
            image_target_mask = torch.zeros_like(image_mask)

        # randomly mask instruction tokens
        if self._dataset._masked_language:
            instr_token, instr_target = randomize_tokens(
                instr_token, instr_mask, self._dataset._tokenizer
            )
        else:
            instr_target = torch.ones_like(instr_token) * -1

        # add perturbations to existing tensors
        image_features = torch.cat([image_features, image_feature])
        image_boxes = torch.cat([image_boxes, image_box])
        image_masks = torch.cat([image_masks.bool(), image_mask.bool()])
        image_targets = torch.cat([image_targets, image_target])
        image_target_masks = torch.cat([image_target_masks, image_target_mask])
        instr_tokens = torch.cat([instr_tokens, instr_token])
        instr_masks = torch.cat([instr_masks, instr_mask])
        instr_targets = torch.cat([instr_targets, instr_target])
        instr_highlights = torch.cat([instr_highlights, instr_highlight])
        segment_ids = torch.cat([segment_ids, segment_id])
        opt_masks = torch.cat([opt_masks, opt_mask]).bool()

        return (
            target,
            image_features,
            image_boxes,
            image_masks,
            image_targets,
            image_target_masks,
            instr_tokens,
            instr_masks,
            instr_targets,
            instr_highlights,
            segment_ids,
            co_attention_masks,
            instr_id,
            opt_masks,
        )

    def _get_ground_truth_trajectory(
        self, beam_index: int
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vln_index = self._dataset._beam_to_vln[beam_index]
        scan_id = self._dataset._vln_data[vln_index]["scan"]
        heading = self._dataset._vln_data[vln_index]["heading"]
        path = self._dataset._vln_data[vln_index]["path"]

        if not self._shortest_path:
            beam_paths = []
            for ranked_path in self._dataset._beam_data[beam_index]["ranked_paths"]:
                beam_paths.append([p for p, _, _ in ranked_path])
            success = self._dataset._get_path_success(scan_id, path, beam_paths)
            if np.sum(success == 1) > 0:
                idx = np.random.choice(np.where(success == 1)[0])  # type: ignore
                path = beam_paths[idx]

        f, b, p, m = self._dataset._get_path_features(scan_id, path, heading)
        image_feature = torch.tensor([f]).float()
        image_box = torch.tensor([b]).float()
        image_prob = torch.tensor([p]).float()
        image_mask = torch.tensor([m]).long()
        return image_feature, image_box, image_prob, image_mask
