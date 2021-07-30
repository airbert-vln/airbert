import logging
from pathlib import Path
from typing import List,  Union, Tuple,  Dict
import torch
from transformers import BertTokenizer

from utils.dataset.common import (
    load_json_data,
    save_json_data,
    tokenize,
)
from utils.dataset.bnb_features_reader import BnBFeaturesReader
from utils.dataset.bnb_dataset import BnBDataset

logger = logging.getLogger(__name__)


def load_tokens_from_trajectories(
    path: Union[Path, str], tokenizer: BertTokenizer, max_instruction_length: int
) -> List[Dict]:
    ppath = Path(path)
    assert ppath.suffix == ".json", ppath

    # load and tokenize data (with caching)
    tokenized_path = (
        ppath.parent / f"{ppath.stem}_tokenized_{max_instruction_length}{ppath.suffix}"
    )

    if tokenized_path.is_file():
        data = load_json_data(tokenized_path)
    else:
        data = load_json_data(ppath)
        for item in data:
            tokenize(item, tokenizer, max_instruction_length)
        save_json_data(data, tokenized_path)
    return data


class BnBSpeakDataset(BnBDataset):
    def __init__(
        self,
        trajectory_path: Union[Path, str],
        tokenizer: BertTokenizer,
        features_reader: BnBFeaturesReader,
        max_instruction_length: int,
        max_length: int,
        max_num_boxes: int,
        default_gpu: bool,
        separators: Tuple[str, ...] = tuple(["[SEP]", ",", "."]),
        split_id: int = 0,
        num_splits: int = 1,
    ):
        self._tokenizer = tokenizer
        self._cls, self._pad, self._sep = self._tokenizer.convert_tokens_to_ids(["[CLS]", "[PAD]", "[SEP]"])  # type: ignore
        if separators:
            self._separators: List[int] = self._tokenizer.convert_tokens_to_ids(separators)  # type: ignore
        else:
            self._separators = [self._sep]

        self._trajectories = load_tokens_from_trajectories(
            trajectory_path, tokenizer, max_instruction_length
        )
        self._samples = []
        for i, samples in enumerate(self._trajectories):
            for j, sample in enumerate(samples):
                sample["instr_id"] = f"{i}-{j}"
                self._samples.append(sample)
        self._samples = self._samples[split_id:: max(1, num_splits)]
        self._features_reader = features_reader
        self._max_instruction_length = max_instruction_length
        self._max_length = max_length
        self._max_num_boxes = max_num_boxes
        self._default_gpu = default_gpu

    def __len__(self):
        return len(self._samples)

    def _remove_special_tokens(self, tokens: List[int]) -> List[int]:
        end = tokens.index(self._pad) - 1 if self._pad in tokens else len(tokens)
        while tokens[end - 1] in self._separators:
            end -= 1
            if end < 0:
                raise ValueError(f"Issue with tokens {tokens}")
        return tokens[1:end]

    def __getitem__(self, sample_index: int):
        sample = self._samples[sample_index]

        # this dataset is used only during testing.
        # so, we don't need to provide any real instruction
        instr_tokens = torch.tensor(sample["instruction_tokens"][0]).long()
        target_tokens = torch.tensor([])
        target_mask = torch.tensor([])
        segment_ids = torch.ones_like(instr_tokens).long()

        # add padding tokens for the generated instruction
        instr_tokens = torch.cat([instr_tokens, torch.tensor([self._pad] * self._max_instruction_length).long()])
        segment_ids = torch.cat([segment_ids, torch.zeros(self._max_instruction_length).long()])
        instr_mask = (instr_tokens > 0).long()

        # get path features
        f, b, _, m = self._get_visual_features(sample["trajectory"])
        image_features = torch.tensor(f).float()
        image_boxes = torch.tensor(b).float()
        image_masks = torch.tensor(m).long()

        # construct null return items
        co_attention_mask = torch.zeros(
            2, self._max_length * self._max_num_boxes, self._max_instruction_length
        ).long()

        i, j = map(int, sample["instr_id"].split("-"))
        instr_id = torch.tensor([i, j]).long()

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
            "instr_id": instr_id,
        }
