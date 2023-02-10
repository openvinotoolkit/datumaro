# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum, auto


class MvtecTask(Enum):
    classification = auto()
    segmentation = auto()
    detection = auto()


class MvtecPath:
    MASK_DIR = "ground_truth"
    MASK_POSTFIX = "_mask"
    IMAGE_EXT = ".png"
    MASK_EXT = ".png"


MvtecCategories = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]
