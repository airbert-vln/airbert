import logging
import copy
import itertools
from dataclasses import dataclass
from itertools import groupby
import random
from typing import (
    List,
    Union,
    Tuple,
    Dict,
    TypeVar,
    Iterator,
    Optional,
    Callable,
    Iterable
)
from operator import itemgetter
import numpy as np
from pathlib import Path
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
from utils.dataset.common import (
    load_json_data, randomize_regions,
    randomize_tokens,
    load_tokens,
)
from utils.dataset.bnb_features_reader import (
    BnBFeaturesReader,
    Trajectory,
    PhotoId,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def shuffle_two(seq: List[T]) -> Iterator[List[T]]:
    n = len(seq)
    ij = list(itertools.permutations(range(n), 2))
    random.shuffle(ij)
    for i, j in ij:
        seq2 = copy.deepcopy(seq)
        seq2[i], seq2[j] = seq2[j], seq2[i]
        yield seq2


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

def is_captionless(photo_id: PhotoId, photo_id_to_caption: Dict[int, Dict]):
    if isinstance(photo_id, (list, tuple)):
        return all(is_captionless(pid, photo_id_to_caption) for pid in photo_id)
    caption = photo_id_to_caption[photo_id]
    return sum(caption["instruction_tokens"][0]) < 204

def random_fill(seq: List[T], fillers: List[T]) -> None:
    n = len(seq)
    random.shuffle(fillers)
    for x in fillers:
        seq.insert(random.randint(0, n - 1), x)
        n += 1


def random_image(listing_ids: List[int], photos_by_listing: Dict[int, List[PhotoId]]) -> Tuple[int, PhotoId]:
    l = random.choice(listing_ids)
    p = random.choice(photos_by_listing[l])
    return l, p



def generate_trajectory_out_listing(
    listing_id: int,
    listing_ids: List[int],
    photos_by_listing: Dict[int, List[PhotoId]],
    photo_id_to_caption: Dict[int, Dict],
    min_length: int = 4,
    max_length: int = 7,
    min_captioned: int = 2,
    max_captioned: int = 7,
    ) -> Tuple[Trajectory, List[bool]]:
    """
    This function is set aside in order to be used by bnb_dataset/scripts/generate_photo_ids
    """
    # Gather all candidates
    path_len = random.randint(min_length, max_length)
    num_captioned = random.randint(min(min_captioned, path_len), min(max_captioned, path_len))
    assert num_captioned > 1
    num_captionless = path_len - num_captioned
    
    captioned: Trajectory = []
    captionless: Trajectory = []
    while len(captioned) < num_captioned or len(captionless) < num_captionless:
        listing_id, photo_id = random_image(listing_ids, photos_by_listing)
        if is_captionless(photo_id, photo_id_to_caption):
            if len(captionless) < num_captionless:
                captionless.append((listing_id, photo_id))
        else:
            if len(captioned) < num_captioned:
                captioned.append((listing_id, photo_id))

    candidates: Trajectory = captioned + captionless
    states: List[bool] = [True] * num_captioned + [False] * num_captionless

    together = list(zip(candidates, states))
    random.shuffle(together)
    candidates, states = list(zip(*together)) # type: ignore

    return candidates, states


def generate_trajectory_from_listing(
    listing_id: int,
    listing_ids: List[int],
    photos_by_listing: Dict[int, List[PhotoId]],
    photo_id_to_caption: Dict[int, Dict],
    min_length: int = 4,
    max_length: int = 7,
    min_captioned: int = 2,
    max_captioned: int = 7,
    ) -> Tuple[Trajectory, List[bool]]:
    """
    This function is set aside in order to be used by bnb_dataset/scripts/generate_photo_ids
    """
    # Gather all candidates
    photo_ids = copy.deepcopy(photos_by_listing[listing_id])
    candidates: Trajectory = [(listing_id, photo_id) for photo_id in photo_ids]
    random.shuffle(candidates)

    # Decide the number of photos
    max_photos = len(candidates)
    path_len = random.randint(min_length, min(max_length, max_photos))

    # Separe captioned from captionless
    states: List[bool] = [not is_captionless(photo_id, photo_id_to_caption) for _, photo_id in candidates]
    captioned_ids, captionless_ids = [], []
    for i, caption in enumerate(states):
        if caption:
            captioned_ids.append(candidates[i])
        else:
            captionless_ids.append(candidates[i])

    # Take a certain number of captioned images, then fill with captionless photo
    # and then with captioned photos
    assert len(captioned_ids) > 1, listing_id
    max_captioned = min(max_captioned, len(captioned_ids), path_len)
    min_captioned = min(min_captioned, len(captioned_ids), path_len)
    assert max_captioned >= min_captioned, (len(captioned_ids), listing_id)
    num_captioned = random.randint(min_captioned, max_captioned)
    candidates = captioned_ids[:num_captioned]
    states = [True] * num_captioned
    candidates += captionless_ids[:path_len - num_captioned]
    states += [False] * (len(candidates) - num_captioned)
    num_captioned2 = max(0, path_len - len(candidates))
    candidates += captioned_ids[num_captioned: num_captioned2 + num_captioned]
    states += [True] * num_captioned2

    # Shuffle again
    together = list(zip(candidates, states))
    random.shuffle(together)
    candidates, states = list(zip(*together)) # type: ignore

    return candidates, states


def generate_negative_trajectories(
    positive_path: Trajectory,
    states: List[bool],
    listing_ids: List[int],
    photos_by_listing: Dict[int, List[PhotoId]],
    photo_id_to_caption: Dict[int, Dict],
    num_negatives: int,
    shuffler: Callable,
):
    path_len = len(positive_path)

    # Create shuffling of captioned images
    captioned_idx: List[int] =[]
    captionless_ids: Trajectory = []
    for i, (sample, state) in enumerate(zip(positive_path, states)):
        if state:
            captioned_idx.append(i)
        else:
            captionless_ids.append(sample)
    shuffled_idx = [n for _, n in zip(range(num_negatives * 2), shuffler(captioned_idx))]

    # Fill these shufflings to create the negative pairs
    negative_captions: List[Trajectory] = []
    for _ in range(num_negatives):
        negative = random.choice(shuffled_idx)
        negative_traj = [positive_path[n] for n in negative]
        random_fill(negative_traj, captionless_ids)
        negative_captions.append(negative_traj)

    negative_images: List[Trajectory] = []
    for _ in range(num_negatives):
        negative = random.choice(shuffled_idx)
        negative_traj = [positive_path[n] for n in negative]
        random_fill(negative_traj, captionless_ids)
        negative_captions.append(negative_traj)

    # Make sure to have at least 1 randomized element with a caption
    negative_randoms: List[Trajectory] = []
    num_flipped = random.randint(1, path_len - 1)
    flipped_idx = list(range(path_len))
    random.shuffle(flipped_idx)
    flipped_idx = flipped_idx[:num_flipped]

    for _ in range(num_negatives):
        path = []
        for i in range(path_len):
            if i in flipped_idx:
                lid, pid = random_image(listing_ids, photos_by_listing)
                found = not is_captionless(pid, photo_id_to_caption)
                while not found:
                    lid, pid = random_image(listing_ids, photos_by_listing)
                    found = not is_captionless(pid, photo_id_to_caption)
            else:
                lid, pid = positive_path[i]
            path.append((lid, pid))
        negative_randoms.append(path)

    return negative_captions, negative_images, negative_randoms

def merge_images(captions: Iterable[Dict]) -> List[PhotoId]:
    return list({
        tuple(p["merging"]) if "merging" in p and len(p["merging"]) > 1 
        else p["photo_id"] 
        for p in captions
    })

def get_key(listing_id, photo_id):
    return f"{listing_id}-{photo_id}"

def _check_in_lmdb(photo_ids_by_listing, keys):
    for listing_id, photo_ids in photo_ids_by_listing.items():
        for photo_id in photo_ids:
            if not isinstance(photo_id, (tuple, list)):
                photo_id = (photo_id, )
            for pid in photo_id:
                if get_key(listing_id, pid) not in keys:
                    raise ValueError(f"{pid, listing_id} is not the LMDB features")
        
def _check_enough_images(photo_ids_by_listing, min_length):
    for listing_id, photo_ids in photo_ids_by_listing.items():
        if len(photo_ids) < min_length:
            raise ValueError(f"Not enough images for listing {listing_id}")

def load_shuffler(shuffler):
    if shuffler == "different":
        return shuffle_different
    elif shuffler == "nonadj":
        return shuffle_non_adjacent
    elif shuffler == "two":
        return shuffle_two
    raise ValueError(f"Unexpected shuffling mode ({shuffler})")

def get_caption(photo_id: PhotoId, photo_id_to_caption: Dict[int, Dict]) -> List[int]:
    # We have a merged image. We pick a caption based on the Places365 score
    if isinstance(photo_id, (tuple, list)):
        # empty photo id (for mypy)
        if not photo_id:  
            raise ValueError("empty photo id")

        # select an image having a caption
        pid = None
        for pid in photo_id:
            if pid in photo_id_to_caption:
                break
        if pid is None:
            return []

        candidates = list(photo_id_to_caption[pid]["merging"])
        weights = list(photo_id_to_caption[pid]["weights"])

        # We consider only candidates having a caption
        for i, candidate in enumerate(candidates):
            if candidate not in photo_id_to_caption:
                weights[i] = 0

        photo_id = int(random.choices(candidates, weights=weights)[0])

    return photo_id_to_caption[photo_id]["instruction_tokens"][0]

def load_trajectories(testset_path: Union[Path, str]):
    testset = load_json_data(testset_path)
    # we convert keys into int
    return {int(key): seq for key, seq in testset.items()}

def _load_skeletons(
        skeleton_path: Union[Path, str], 
        tokenizer,
        max_instruction_length: int 
        ) -> Optional[Dict[int, List[Dict]]]:
    skeletons = load_tokens(skeleton_path, tokenizer, max_instruction_length)
    skeletons = sorted(skeletons, key=lambda s: sum(s["np"]))
    skeletons_by_length = {length: list(s) for length, s in groupby(skeletons, key=lambda s: sum(s["np"]))}
    return skeletons_by_length


class InstructionGenerator:
    """
    Given a trajectory, it can generate an instruction
    """
    def __init__(self, tokenizer: BertTokenizer, separators: Tuple[str, ...], photo_id_to_caption: Dict[int, Dict], max_instruction_length: int):
        self._tokenizer = tokenizer
        self._cls, self._pad, self._sep = self._tokenizer.convert_tokens_to_ids(["[CLS]", "[PAD]", "[SEP]"])  # type: ignore

        if separators:
            self._separators: List[Optional[int]] = []
            seps: List[str] = list(separators)
            while None in seps:
                seps = seps.pop(seps.index(None)) # type: ignore
                self._separators.append(None)
            self._separators += self._tokenizer.convert_tokens_to_ids(seps) # type: ignore
        else:
            self._separators = [self._sep]

        self._max_instruction_length = max_instruction_length
        self._photo_id_to_caption = photo_id_to_caption

    def _remove_special_tokens(self, tokens: List[int]) -> List[int]:
        end = tokens.index(self._pad) - 1 if self._pad in tokens else len(tokens)
        while tokens[end - 1] in self._separators:
            end -= 1
            if end < 0:
                raise ValueError(f"Issue with tokens {tokens}")
        return tokens[1: end]


    def __call__(self,  trajectory: Trajectory) -> List[int]:
        raise NotImplementedError()


class RephraseInstructionGenerator(InstructionGenerator):
    """
    Fill the blanks on a R2R instruction using NP from Airbnb
    """
    def __init__(self, skeleton_path: Union[Path, str],   *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skeleton_path = skeleton_path
        self._skeletons_by_length = _load_skeletons(self._skeleton_path, self._tokenizer, self._max_instruction_length)
        

    def __call__(self, trajectory: Trajectory) -> List[int]:
        # gather captions
        captions: List[List[int]] = []
        photo_id: PhotoId
        for _, photo_id in trajectory:
            if is_captionless(photo_id, self._photo_id_to_caption):
                continue
            caption = get_caption(photo_id, self._photo_id_to_caption)
            caption = self._remove_special_tokens(caption)
            captions.append(caption)

        # pick a skeleton
        if self._skeletons_by_length is None:
            raise ValueError("Should not happen")

        # fill the skeleton
        skeleton = random.choice(self._skeletons_by_length[len(captions)])
        sentence = [self._cls]
        counter = 0
        for np, split in zip(skeleton["np"], skeleton["instruction_tokens"]):
            if np:
                caption = captions[counter]
                counter += 1
            else:
                caption = self._remove_special_tokens(split)
            sentence += caption

        sentence = sentence[:self._max_instruction_length - 1]
        sentence += [self._sep] 
        sentence += [self._pad] * (self._max_instruction_length - len(sentence))

        return sentence

class ConcatenateInstructionGenerator(InstructionGenerator):
    """
    Contenate captions in order to create a fake instruction
    """
    @property
    def sep(self) -> List[int]:
        """ Select a separator token """
        _sep = random.choice(self._separators)
        if _sep is not None:
            return [_sep]
        return []

    def __call__(self, trajectory: Trajectory) -> List[int]:
        # gather captions
        captions: List[List[int]] = []
        photo_id: PhotoId
        for _, photo_id in trajectory:
            if is_captionless(photo_id, self._photo_id_to_caption):
                continue
            caption = get_caption(photo_id, self._photo_id_to_caption)
            caption = self._remove_special_tokens(caption)
            captions.append(caption)
        
        # shorten some captions
        credit = self._max_instruction_length
        credit -= 1 # CLS token
        credit -= len(captions) # connector
        quota = credit // len(captions)
        exceeding_ids = []
        exceeding_lengths = []
        for idx, caption in enumerate(captions):
            num_tokens = len(caption)
            if num_tokens > quota:
                exceeding_ids.append(idx)
                exceeding_lengths.append(num_tokens)
            else:
                credit -= num_tokens

        if exceeding_ids != []:
            exceeding_lengths, exceeding_ids = list(zip(*sorted(zip(exceeding_lengths, exceeding_ids)))) # type: ignore
            for i, idx in enumerate(exceeding_ids):
                num_tokens = credit // len(exceeding_ids[i:])
                captions[idx] = captions[idx][:num_tokens]
                credit -= len(captions[idx])
                assert credit >= 0

        # concatenate with separators
        merge: List[int] = [self._cls]
        for i, caption in enumerate(captions):
            merge += caption
            if i < len(captions) - 1:
                merge += self.sep
        merge += [self._sep]
        
        # pad sentence
        merge += [self._pad] * (self._max_instruction_length - len(merge))
        
        return merge


class BnBDataset(Dataset):
    def __init__(
        self,
        caption_path: Union[Path, str],
        testset_path: Union[Path, str],
        tokenizer: BertTokenizer,
        features_reader: BnBFeaturesReader,
        max_instruction_length: int,
        max_num_boxes: int,
        min_length: int,
        max_length: int,
        min_captioned: int,
        max_captioned: int,
        num_negatives: int,
        num_positives: int,
        masked_vision: bool,
        masked_language: bool,
        skeleton_path: Union[Path, str] = "",
        highlighted_language: bool = False,
        training: bool = False,
        out_listing: bool = False,
        shuffler: str = "different",
        separators: Tuple[str, ...] = tuple(),
    ):
        self._tokenizer = tokenizer

        captions = load_tokens(caption_path, tokenizer, max_instruction_length)
        self._photo_id_to_caption = {
            int(caption["photo_id"]): caption for caption in captions
        }
        captions = sorted(captions, key=itemgetter("listing_id"))
        
        # gather photo_ids by listing
        self._photo_ids_by_listing = {
            listing: merge_images(photos)
            for listing, photos in groupby(captions, key=itemgetter("listing_id"))
        }
        self._listing_ids = list(self._photo_ids_by_listing.keys())
        
        # WARNING: make sure that the photo ids do not refer to a missing LMDB feature
        _check_in_lmdb(self._photo_ids_by_listing, set(features_reader.keys))
        
        # WARNING: Make sure the listing contains enough images
        if not out_listing:
            _check_enough_images(self._photo_ids_by_listing, min_length)
        
        self._build_instructions: List[InstructionGenerator] = []

        if skeleton_path == "":
            self._build_instructions.append(
                ConcatenateInstructionGenerator(
                    tokenizer=tokenizer,
                    separators=separators,
                    photo_id_to_caption=self._photo_id_to_caption,
                    max_instruction_length=max_instruction_length,
                )
            )
        else:
            self._build_instructions.append(RephraseInstructionGenerator(
                skeleton_path=skeleton_path,
                tokenizer=tokenizer,
                separators=separators,
                photo_id_to_caption=self._photo_id_to_caption,
                max_instruction_length=max_instruction_length,
            )
            )

        self._testset = load_trajectories(testset_path) if not training else {}
        self._shuffler = load_shuffler(shuffler)
        self._features_reader = features_reader
        self._out_listing = out_listing
        self._max_instruction_length = max_instruction_length
        self._max_num_boxes = max_num_boxes
        self._max_length = max_length
        self._min_length = min_length
        self._max_captioned = max_captioned
        self._min_captioned = min_captioned
        self._num_positives = num_positives
        self._num_negatives = num_negatives
        self._masked_vision = masked_vision
        self._masked_language = masked_language
        self._training = training
        self._highlighted_language = highlighted_language
        if self._highlighted_language:
            raise NotImplementedError()

    def __len__(self):
        if self._out_listing:
            threshold = 15000 if self._training else 500
            return min(len(self._listing_ids), threshold)
        else:
            return len(self._listing_ids)

    def _pick_photo_ids(
        self, listing_id: int
    ) -> Tuple[Trajectory, List[Trajectory], List[Trajectory], List[Trajectory]]:
        if not self._training:
            return self._testset[listing_id]
        
        fn = generate_trajectory_from_listing if not self._out_listing else generate_trajectory_out_listing

        positive_trajectory, captioned = fn(
            listing_id,
            self._listing_ids,
            self._photo_ids_by_listing,
            self._photo_id_to_caption,
            self._min_length,
            self._max_length,
            self._min_captioned,
            self._max_captioned,
        )

        neg_captions, neg_images, neg_randoms = generate_negative_trajectories(
            positive_trajectory,
            captioned,
            self._listing_ids,
            self._photo_ids_by_listing,
            self._photo_id_to_caption,
            self._num_negatives,
            shuffler=self._shuffler,
        )

        if self._out_listing:
            neg_randoms = []

        return positive_trajectory, neg_captions, neg_images, neg_randoms

    def __getitem__(self, index: int):
        listing_id = self._listing_ids[index]

        # select negative and positive photo ids
        (
            positive_ids,
            negative_captions,
            negative_images,
            negative_random,
        ) = self._pick_photo_ids(listing_id)


        # get the positive pair
        build_instruction = random.choice(self._build_instructions)
        instructions = [build_instruction(positive_ids)]
        f, b, p, m = self._get_visual_features(positive_ids)
        features, boxes, probs, masks = [f], [b], [p], [m]

        # get the negative captions
        for traj in negative_captions:
            instructions += [build_instruction(traj)]
            features += [features[0]]
            boxes += [boxes[0]]
            probs += [probs[0]]
            masks += [masks[0]]

        # get the negative images
        for traj in negative_images:
            instructions += [instructions[0]]
            f, b, p, m = self._get_visual_features(traj)
            features += [f]
            boxes += [b]
            probs += [p]
            masks += [m]

        # get the random images
        for traj in negative_random:
            instructions += [instructions[0]]
            f, b, p, m = self._get_visual_features(traj)
            features += [f]
            boxes += [b]
            probs += [p]
            masks += [m]

        # convert data into tensors
        image_features = torch.tensor(features).float()
        image_boxes = torch.tensor(boxes).float()
        image_probs = torch.tensor(probs).float()
        image_masks = torch.tensor(masks).long()
        instr_tokens = torch.tensor(instructions).long()
        instr_mask = instr_tokens > 0
        segment_ids = torch.zeros_like(instr_tokens)
        instr_highlights = torch.zeros((image_features.shape[0], 0)).long()

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
            2, self._max_length * self._max_num_boxes, self._max_instruction_length
        ).long()

        if self._training:
            target = torch.tensor(0)
        else:
            target = torch.zeros(image_features.shape[0]).bool()
            target[0] = 1

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
            torch.tensor(listing_id).long(),
            torch.ones(image_features.shape[0]).bool(),
        )

    def _get_visual_features(self, trajectory: Trajectory):
        """ Get features for a given path. """
        path_length = min(len(trajectory), self._max_length)
        path_features, path_boxes, path_probs, path_masks = [], [], [], []
        for i, (listing_id, photo_id) in enumerate(trajectory):
            # get image features
            if isinstance(photo_id, int):
                photo_id = tuple([photo_id])
            keys = tuple(f"{listing_id}-{pid}" for pid in photo_id)
            features, boxes, probs = self._features_reader[keys]

            num_boxes = min(len(boxes), self._max_num_boxes)

            # pad features and boxes (if needed)
            pad_features = np.zeros((self._max_num_boxes, 2048))
            pad_features[:num_boxes] = features[:num_boxes]

            pad_boxes = np.zeros((self._max_num_boxes, 12))
            pad_boxes[:num_boxes, :11] = boxes[:num_boxes, :11]  # type: ignore
            pad_boxes[:, 11] = np.ones(self._max_num_boxes) * i

            pad_probs = np.zeros((self._max_num_boxes, 1601))
            pad_probs[:num_boxes] = probs[:num_boxes]

            box_pad_length = self._max_num_boxes - num_boxes
            pad_masks = [1] * num_boxes + [0] * box_pad_length

            path_features.append(pad_features)
            path_boxes.append(pad_boxes)
            path_probs.append(pad_probs)
            path_masks.append(pad_masks)

        # pad path lists (if needed)
        for path_idx in range(path_length, self._max_length):
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
