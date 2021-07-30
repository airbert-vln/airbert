import base64
from typing import Tuple, Dict, Set
import pickle
import numpy as np
from .features_reader import FeaturesReader


def _convert_item(item):
    # item['scanId'] is unchanged
    # item['viewpointId'] is unchanged
    item["image_w"] = int(item["image_w"])  # pixels
    item["image_h"] = int(item["image_h"])  # pixels
    item["vfov"] = int(item["vfov"])  # degrees
    item["features"] = np.frombuffer(
        base64.b64decode(item["features"]), dtype=np.float32
    ).reshape(
        (-1, 2048)
    )  # K x 2048 region features
    item["boxes"] = np.frombuffer(
        base64.b64decode(item["boxes"]), dtype=np.float32
    ).reshape(
        (-1, 4)
    )  # K x 4 region coordinates (x1, y1, x2, y2)
    item["cls_prob"] = np.frombuffer(
        base64.b64decode(item["cls_prob"]), dtype=np.float32
    ).reshape(
        (-1, 1601)
    )  # K x 1601 region object class probabilities
    # item["attr_prob"] = np.frombuffer(
    #     base64.b64decode(item["attr_prob"]), dtype=np.float32
    # ).reshape(
    #     (-1, 401)
    # )  # K x 401 region attribute class probabilities
    item["viewHeading"] = np.frombuffer(
        base64.b64decode(item["viewHeading"]), dtype=np.float32
    )  # 36 values (heading of each image)
    item["viewElevation"] = np.frombuffer(
        base64.b64decode(item["viewElevation"]), dtype=np.float32
    )  # 36 values (elevation of each image)
    item["featureHeading"] = np.frombuffer(
        base64.b64decode(item["featureHeading"]), dtype=np.float32
    )  # K headings for the features
    item["featureElevation"] = np.frombuffer(
        base64.b64decode(item["featureElevation"]), dtype=np.float32
    )  # K elevations for the features
    item["featureViewIndex"] = np.frombuffer(
        base64.b64decode(item["featureViewIndex"]), dtype=np.float32
    )  # K indices mapping each feature to one of the 36 images


def _get_boxes(item):
    image_width = item["image_w"]
    image_height = item["image_h"]

    boxes = item["boxes"]
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


def _get_locations(boxes, feat_headings, feat_elevations, heading, next_heading):
    """ Convert boxes and orientation information into locations. """
    N = len(boxes)
    locations = np.ones(shape=(N, 11), dtype=np.float32)

    # region encoding
    locations[:, 0] = boxes[:, 0]
    locations[:, 1] = boxes[:, 1]
    locations[:, 2] = boxes[:, 2]
    locations[:, 3] = boxes[:, 3]
    locations[:, 4] = boxes[:, 4]

    # orientation encoding
    locations[:, 5] = np.sin(feat_headings - heading)
    locations[:, 6] = np.cos(feat_headings - heading)
    locations[:, 7] = np.sin(feat_elevations)
    locations[:, 8] = np.cos(feat_elevations)

    # next orientation encoding
    locations[:, 9] = np.sin(feat_headings - next_heading)
    locations[:, 10] = np.cos(feat_headings - next_heading)

    return locations


class PanoFeaturesReader(FeaturesReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # get viewpoints
        self.viewpoints: Dict[str, Set[str]] = {}
        for key in self.keys:
            scan_id, viewpoint_id = key.split("-")
            if scan_id not in self.viewpoints:
                self.viewpoints[scan_id] = set()
            self.viewpoints[scan_id].add(viewpoint_id)

    def __getitem__(self, query: Tuple):
        key, heading, next_heading = query  # unpack key
        if key not in self.keys:
            raise TypeError(f"invalid key: {key}")

        env = self.envs[self.keys[key]]
        # load from disk
        with env.begin(write=False) as txn:
            item = pickle.loads(txn.get(key.encode()))  # type: ignore
            _convert_item(item)

            boxes = _get_boxes(item)
            probs = item["cls_prob"]
            features = item["features"]
            headings = item["featureHeading"]
            elevations = item["featureElevation"]

        if not isinstance(features, np.ndarray):
            raise RuntimeError(f"Unexpected type for features ({type(features)})")

        locations = _get_locations(boxes, headings, elevations, heading, next_heading)

        # add a global feature vector
        g_feature = features.mean(axis=0, keepdims=True)
        g_location = np.array(
            [
                [
                    0,
                    0,
                    1,
                    1,
                    1,
                    np.sin(0 - heading),
                    np.cos(0 - heading),
                    np.sin(0),
                    np.cos(0),
                    np.sin(0 - next_heading),
                    np.cos(0 - next_heading),
                ]
            ]
        )
        g_prob = np.ones(shape=(1, 1601)) / 1601  # uniform probability

        features = np.concatenate([g_feature, features], axis=0)
        locations = np.concatenate([g_location, locations], axis=0)
        probs = np.concatenate([g_prob, probs], axis=0)

        return features, locations, probs
