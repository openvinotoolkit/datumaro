# Copyright (C) 2019-2022 Intel Corporation
# Copyright (C) 2022 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os.path as osp
import sys
from collections import OrderedDict
from enum import Enum, auto
from itertools import chain
from typing import Dict, Optional, Set, Union

import numpy as np
from attr import define, field

from datumaro.components.annotation import (
    AnnotationType,
    Colormap,
    LabelCategories,
    MaskCategories,
    RgbColor,
)
from datumaro.components.errors import InvalidAnnotationError
from datumaro.components.extractor import CategoriesInfo
from datumaro.util import dump_json_file, find, parse_json_file
from datumaro.util.meta_file_util import get_meta_file


class VocTask(Enum):
    classification = auto()
    detection = auto()
    segmentation = auto()
    action_classification = auto()
    person_layout = auto()


class VocLabel(Enum):
    background = 0
    aeroplane = 1
    bicycle = 2
    bird = 3
    boat = 4
    bottle = 5
    bus = 6
    car = 7
    cat = 8
    chair = 9
    cow = 10
    diningtable = 11
    dog = 12
    horse = 13
    motorbike = 14
    person = 15
    pottedplant = 16
    sheep = 17
    sofa = 18
    train = 19
    tvmonitor = 20
    ignored = 255


class VocPose(Enum):
    Unspecified = auto()
    Left = auto()
    Right = auto()
    Frontal = auto()
    Rear = auto()


class VocBodyPart(Enum):
    head = auto()
    hand = auto()
    foot = auto()


class VocAction(Enum):
    other = auto()
    jumping = auto()
    phoning = auto()
    playinginstrument = auto()
    reading = auto()
    ridingbike = auto()
    ridinghorse = auto()
    running = auto()
    takingphoto = auto()
    usingcomputer = auto()
    walking = auto()


def generate_colormap(length: int = 256) -> Colormap:
    def get_bit(number, index):
        return (number >> index) & 1

    colormap = np.zeros((length, 3), dtype=int)
    indices = np.arange(length, dtype=int)

    for j in range(7, -1, -1):
        for c in range(3):
            colormap[:, c] |= get_bit(indices, c) << j
        indices >>= 3

    return OrderedDict((id, tuple(color)) for id, color in enumerate(colormap))


VocColormap: Colormap = {
    id: color for id, color in generate_colormap(256).items() if id in {l.value for l in VocLabel}
}
VocInstColormap = generate_colormap(256)


class VocPath:
    IMAGES_DIR = "JPEGImages"
    ANNOTATIONS_DIR = "Annotations"
    SEGMENTATION_DIR = "SegmentationClass"
    INSTANCES_DIR = "SegmentationObject"
    SUBSETS_DIR = "ImageSets"
    IMAGE_EXT = ".jpg"
    SEGM_EXT = ".png"
    LABELMAP_FILE = "labelmap.txt"

    TASK_DIR = {
        VocTask.classification: "Main",
        VocTask.detection: "Main",
        VocTask.segmentation: "Segmentation",
        VocTask.action_classification: "Action",
        VocTask.person_layout: "Layout",
    }


@define
class VocLabelInfo:
    color: Optional[RgbColor] = None
    parts: Set[str] = field(factory=set, converter=set)
    actions: Set[str] = field(factory=set, converter=set)


if sys.version_info < (3, 9):
    _voc_label_map_base = OrderedDict
else:
    _voc_label_map_base = OrderedDict[str, VocLabelInfo]


class VocLabelMap(_voc_label_map_base):
    """
    Provides VOC-specific info about labels.
    A mapping of type: str -> VocLabelInfo
    """

    DEFAULT_BACKGROUND_LABEL = "background"
    DEFAULT_BACKGROUND_COLOR: RgbColor = (0, 0, 0)

    def __init__(self, *args, **kwargs) -> None:
        d = OrderedDict(*args, **kwargs)

        super().__init__(
            (k, v if isinstance(v, VocLabelInfo) else VocLabelInfo(*v)) for k, v in d.items()
        )

    def is_label(self, label: str) -> bool:
        return self.get(label) is not None

    def is_part(self, part: str) -> bool:
        for label_desc in self.values():
            if part in label_desc.parts:
                return True
        return False

    def is_action(self, label: str, action: str) -> bool:
        return action in self.get_actions(label)

    def get_actions(self, label: str) -> Set[str]:
        label_desc = self.get(label)
        if not label_desc:
            return set()
        return label_desc.actions

    def has_colors(self) -> bool:
        return any(v.color is not None for v in self.values())

    def find_background_label(
        self,
        *,
        name: str = DEFAULT_BACKGROUND_LABEL,
        color: RgbColor = DEFAULT_BACKGROUND_COLOR,
    ) -> Optional[str]:
        bg_label = find(self.items(), lambda x: x[1].color == color)
        if bg_label is not None:
            return bg_label[0]

        if name in self:
            return name

        return None

    def find_or_create_background_label(
        self,
        *,
        name: str = DEFAULT_BACKGROUND_LABEL,
        color: RgbColor = DEFAULT_BACKGROUND_COLOR,
    ) -> str:
        bg_label = self.find_background_label(color=color, name=name)

        if bg_label is None:
            bg_label = name
            color = color if self.has_colors() else None
            self[bg_label] = VocLabelInfo(color)
            self.move_to_end(bg_label, last=False)

        return bg_label

    @staticmethod
    def make_default() -> VocLabelMap:
        labels = sorted(VocLabel, key=lambda l: l.value)
        label_map = VocLabelMap(
            (label.name, VocLabelInfo(VocColormap[label.value])) for label in labels
        )
        label_map[VocLabel.person.name].parts = [p.name for p in VocBodyPart]
        label_map[VocLabel.person.name].actions = [a.name for a in VocAction]
        return label_map

    @staticmethod
    def from_categories(categories: CategoriesInfo) -> VocLabelMap:
        if AnnotationType.mask not in categories:
            # generate a new colormap for the input labels
            labels = categories.get(AnnotationType.label, LabelCategories())
            label_map = VocLabelMap((item.name, VocLabelInfo()) for item in labels.items)

        else:
            # use the source colormap
            labels: LabelCategories = categories[AnnotationType.label]
            colors: MaskCategories = categories[AnnotationType.mask]
            label_map = VocLabelMap()
            for idx, item in enumerate(labels.items):
                color = colors.colormap.get(idx)
                if color is not None:
                    label_map[item.name] = VocLabelInfo(color)

        return label_map

    @staticmethod
    def parse_from_file(path: str) -> VocLabelMap:
        """
        Parses a label map file in the format:
        'name : color (r, g, b) : parts (hand, feet, ...) : actions (siting, standing, ...)'

        Parameters:
            path: File path

        Returns:
            A dictionary: label -> (color, parts, actions)
        """

        label_map = OrderedDict()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                # skip empty and commented lines
                line = line.strip()
                if not line or line[0] == "#":
                    continue

                # name : color : parts : actions
                label_desc = line.strip().split(":")
                if len(label_desc) != 4:
                    raise InvalidAnnotationError(
                        f"Label description has wrong number of fields '{len(label_desc)}'. "
                        "Expected 4 ':'-separated fields."
                    )
                name = label_desc[0]

                if name in label_map:
                    raise InvalidAnnotationError(
                        f"Label '{name}' is already defined in the label map"
                    )

                if 1 < len(label_desc) and label_desc[1]:
                    color = label_desc[1].split(",")
                    if len(color) != 3:
                        raise InvalidAnnotationError(
                            f"Label '{name}' has wrong color '{color}'. Expected an 'r,g,b' triplet."
                        )
                    color = tuple(int(c) for c in color)
                else:
                    color = None

                if 2 < len(label_desc) and label_desc[2]:
                    parts = [s.strip() for s in label_desc[2].split(",")]
                else:
                    parts = []

                if 3 < len(label_desc) and label_desc[3]:
                    actions = [s.strip() for s in label_desc[3].split(",")]
                else:
                    actions = []

                label_map[name] = [color, parts, actions]
        return VocLabelMap(label_map)

    @staticmethod
    def parse_from_meta_file(path: str) -> VocLabelMap:
        # Uses a custom format with extra fields
        meta_file = path
        if osp.isdir(path):
            meta_file = get_meta_file(path)

        dataset_meta = parse_json_file(meta_file)

        label_map = OrderedDict()
        parts = dataset_meta.get("parts", {})
        actions = dataset_meta.get("actions", {})

        for i, label in enumerate(dataset_meta.get("labels", [])):
            label_map[label] = [None, parts.get(str(i), []), actions.get(str(i), [])]

        colors = dataset_meta.get("segmentation_colors", [])

        for i, label in enumerate(dataset_meta.get("label_map", {}).values()):
            if label not in label_map:
                label_map[label] = [None, [], []]

            if any(colors) and colors[i] is not None:
                label_map[label][0] = tuple(colors[i])

        return VocLabelMap(label_map)

    def dump_to_file(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# label:color_rgb:parts:actions\n")
            for label_name, label_desc in self.items():
                if label_desc.color:
                    color_rgb = ",".join(str(c) for c in label_desc.color)
                else:
                    color_rgb = ""

                parts = ",".join(str(p) for p in label_desc.parts)
                actions = ",".join(str(a) for a in label_desc.actions)

                f.write("%s\n" % ":".join([label_name, color_rgb, parts, actions]))

    def dump_to_meta_file(self, output_dir: str):
        # Uses custom format with extra fields
        dataset_meta = {}

        labels = []
        labels_dict = {}
        segmentation_colors = []
        parts = {}
        actions = {}

        for i, (label_name, label_desc) in enumerate(self.items()):
            labels.append(label_name)
            if label_desc.color:
                labels_dict[str(i)] = label_name
                segmentation_colors.append(list(map(int, label_desc.color)))

            parts[str(i)] = list(label_desc.parts)
            actions[str(i)] = list(label_desc.actions)

        dataset_meta["labels"] = labels

        if any(segmentation_colors):
            dataset_meta["label_map"] = labels_dict
            dataset_meta["segmentation_colors"] = segmentation_colors

            bg_label = self.find_background_label()
            if bg_label:
                dataset_meta["background_label"] = bg_label

        if any(parts):
            dataset_meta["parts"] = parts

        if any(actions):
            dataset_meta["actions"] = actions

        dump_json_file(get_meta_file(output_dir), dataset_meta)


def make_voc_categories(label_map: Optional[Union[VocLabelMap, Dict]] = None) -> CategoriesInfo:
    if label_map is None:
        label_map = VocLabelMap.make_default()
    elif not isinstance(label_map, VocLabelMap):
        label_map = VocLabelMap(label_map)

    categories = {}

    label_categories = LabelCategories()
    label_categories.attributes.update(["difficult", "truncated", "occluded"])

    for label, desc in label_map.items():
        label_categories.add(label, attributes=desc.actions)
    for part in sorted(set(chain(*(desc.parts for desc in label_map.values())))):
        label_categories.add(part)
    categories[AnnotationType.label] = label_categories

    if not label_map.has_colors():  # generate new colors
        colormap = generate_colormap(len(label_map))
    else:  # only copy defined colors
        label_id = lambda label: label_categories.find(label)[0]
        colormap = {
            label_id(name): desc.color for name, desc in label_map.items() if desc.color is not None
        }
    mask_categories = MaskCategories(colormap)
    mask_categories.inverse_colormap  # pylint: disable=pointless-statement
    categories[AnnotationType.mask] = mask_categories

    return categories
