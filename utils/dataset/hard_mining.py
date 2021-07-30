# pylint: disable=no-member, not-callable
"""
Mixin for hard mining examples
"""
import logging
from typing import List, Tuple
from pathlib import Path
import numpy as np
import copy
import torch
from utils.dataset.common import pad_packed
from .beam_dataset import BeamDataset, randomize_regions, randomize_tokens

logger = logging.getLogger(__name__)


def compute_prob(x):
    sig = 1 / (1 + np.exp(-x)) # type: ignore
    sig /= np.sum(sig)
    return sig


class HardMiningDataset(BeamDataset):
    def __init__(self, save_folder: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._instr_id_to_beam = {}
        for i, beam_item in enumerate(self._beam_data):
            self._instr_id_to_beam[beam_item["instr_id"]] = i

        self._save_folder = Path(save_folder)

    @property
    def weights(self):
        if not hasattr(self, "_weights"):
            if not hasattr(self, "__len__"):
                return None
            if (self._save_folder / "hard_mining.pth").is_file():
                self._weights = torch.load(self._save_folder / "hard_mining.pth")
            else:
                num_samples = len(self)
                # Intialize weights with a uniform distribution
                if self._default_gpu:
                    logger.info("Initializing the hard mining weights")
                self._weights = np.zeros((num_samples, 30))
        return self._weights

    def save(self):
        torch.save(self.weights, self._save_folder / "hard_mining.pth")

    def post_step(self, output: Tuple[torch.Tensor, ...], batch: List[torch.Tensor]):
        """
        Using the prediction of the network, we update the weights
        """
        if not self._training:
            return

        meta = batch[12]
        opt_mask: torch.BoolTensor = copy.deepcopy(batch[13]).bool() # type: ignore
        vil_logit = pad_packed(output[0].squeeze(1), opt_mask)
        ground_truth = batch[0]

        for logit, mask, gt, item in zip(vil_logit, opt_mask, ground_truth, meta):  # type: ignore
            if gt == -1:
                continue
            mask[gt] = False

            # remove extra neg samples for perturbations
            mask = mask[: self._num_beams]
            logit = logit[: self._num_beams]

            instr_id = f"{item[0]}_{item[1]}"
            beam_index = self._instr_id_to_beam[instr_id]

            # 0: path_id, 1: stc_id
            cand_index = item[2:][mask].cpu()
            self.weights[beam_index, cand_index] = logit[mask].detach().cpu()

    def __getitem__(self, beam_index: int):
        vln_index = self._beam_to_vln[beam_index]
        vln_item = self._vln_data[vln_index]

        # get beam info
        path_id, instruction_index = map(
            int, self._beam_data[beam_index]["instr_id"].split("_")
        )

        # get vln info
        scan_id = self._vln_data[vln_index]["scan"]
        heading = self._vln_data[vln_index]["heading"]
        gt_path = self._vln_data[vln_index]["path"]

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

        # get the positive and negative examples
        beam_paths = []
        for ranked_path in self._beam_data[beam_index]["ranked_paths"]:
            beam_paths.append([p for p, _, _ in ranked_path])

        success = self._get_path_success(scan_id, gt_path, beam_paths)
        if self._training:
            # select one positive and three negative paths
            if np.sum(success == 1) == 0 or np.sum(success == 0) < self._num_beams - 1:
                # not enough positive or negative paths (this should be rare)
                target = -1  # default ignore index
                selected_index: List[int] = list(range(self._num_beams))
            else:
                target = 0
                selected_index = []
                # first select a positive
                idx = np.random.choice(np.where(success == 1)[0]) # type: ignore
                selected_index.append(idx)
                # next select three negatives
                mask = success == 0
                prob = self.weights[beam_index, : len(beam_paths)][mask]
                prob = compute_prob(prob)
                idxs = np.random.choice(# type: ignore
                    np.arange(len(beam_paths))[mask],
                    size=self._num_beams - 1,
                    p=prob,
                    replace=False,
                    ) 
                selected_index += list(idxs)
        else:
            target = success
            selected_index = list(range(len(beam_paths)))

        selected_paths = [beam_paths[idx] for idx in selected_index]

        # shuffle the visual features from the ground truth as a free negative path
        if self._shuffle_visual_features:
            path = self._vln_data[vln_index]["path"]
            selected_paths += [corr for corr, _ in zip(self._shuffler(path), range(2))]

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
        instr_highlights = instr_highlights.repeat(len(features), 1).long()
        segment_ids = segment_ids.repeat(len(features), 1).long()

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

        # set target
        target = torch.tensor(target).long()  # type: ignore

        # construct null return items
        co_attention_mask = torch.zeros(
            2, self._max_path_length * self._max_num_boxes, self._max_instruction_length
        ).long()
        # we encode the selected path index inside the instr_id + the selected index
        instr_id = torch.tensor([path_id, instruction_index] + selected_index).long()

        return (
            target,
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

