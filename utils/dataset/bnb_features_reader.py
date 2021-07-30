import base64
from dataclasses import dataclass
from typing import DefaultDict, Dict, Tuple, List, Optional, Union, Set
from collections import defaultdict
import pickle
import lmdb
import numpy as np
from .features_reader import FeaturesReader

PhotoId = Union[int, Tuple[int, ...]]
Sample = Tuple[int, PhotoId]  # listing id, photo id
Trajectory = List[Sample]

PhotoIdType = Dict[int, Tuple[Trajectory, List[Trajectory], List[Trajectory], List[Trajectory]]]


@dataclass
class Record:
    photo_id: int
    listing_id: int
    num_boxes: int
    image_width: int
    image_height: int
    cls_prob: np.ndarray
    features: np.ndarray
    boxes: np.ndarray


def _convert_item(key: str, item: Dict) -> Record:
    # FIXME use one convention and not two!!
    photo_id, listing_id = map(int, key.split("-"))

    old = "image_width" in item

    image_w = int(item["image_width" if old else "image_w"])  # pixels
    image_h = int(item["image_height" if old else "image_h"])  # pixels
    features = np.frombuffer(
        item["feature"] if old else base64.b64decode(item["features"]),
        dtype=np.float32,
    )
    features = features.reshape((-1, 2048))  # K x 2048 region features
    boxes = np.frombuffer(
        item["bbox"] if old else base64.b64decode(item["boxes"]), dtype=np.float32,
    )
    boxes = boxes.reshape((-1, 4))  # K x 4 region coordinates (x1, y1, x2, y2)
    num_boxes = int(boxes.shape[0])
    cls_prob = np.frombuffer(
        item["cls_prob"] if old else base64.b64decode(item["cls_prob"]),
        dtype=np.float32,
    )
    cls_prob = cls_prob.reshape(
        (-1, 1601)
    )  # K x 1601 region object class probabilities
    return Record(
        photo_id, listing_id, num_boxes, image_w, image_h, cls_prob, features, boxes,
    )


def _get_boxes(record: Record) -> np.ndarray:
    image_width = record.image_width
    image_height = record.image_height

    boxes = record.boxes
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area /= image_width * image_height

    N = len(boxes)
    output = np.zeros(shape=(N, 5), dtype=np.float32)

    # region encoding
    output[:, 0] = boxes[:, 0] / image_width
    output[:, 1] = boxes[:, 1] / image_height
    output[:, 2] = boxes[:, 2] / image_width
    output[:, 3] = boxes[:, 3] / image_height
    output[:, 4] = area

    return output


def _get_locations(boxes: np.ndarray):
    """ Convert boxes and orientation information into locations. """
    N = len(boxes)
    locations = np.ones(shape=(N, 11), dtype=np.float32)

    # region encoding
    locations[:, 0] = boxes[:, 0]
    locations[:, 1] = boxes[:, 1]
    locations[:, 2] = boxes[:, 2]
    locations[:, 3] = boxes[:, 3]
    locations[:, 4] = boxes[:, 4]

    # other indices are used for Room-to-Room

    return locations


class BnBFeaturesReader(FeaturesReader):

    def __getitem__(self, query: Tuple):
        l_boxes, l_probs, l_features = [], [], []
        items = super().__getitem__(query)
        for key, item in zip(list(query), items):
            record = _convert_item(key, item)
            l_boxes.append(_get_boxes(record))
            l_probs.append(record.cls_prob)
            l_features.append(record.features)

        features: np.ndarray = np.concatenate(l_features, axis=0)
        boxes = np.concatenate(l_boxes, axis=0)
        probs = np.concatenate(l_probs, axis=0)
        locations = _get_locations(boxes)

        if features.size == 0:
            raise RuntimeError("Features could not be correctly read")

        # add a global feature vector
        g_feature = features.mean(axis=0, keepdims=True)
        g_location = np.array([[0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1,]])
        g_prob = np.ones(shape=(1, 1601)) / 1601  # uniform probability

        features = np.concatenate([g_feature, features], axis=0)
        locations = np.concatenate([g_location, locations], axis=0)
        probs = np.concatenate([g_prob, probs], axis=0)

        return features, locations, probs
