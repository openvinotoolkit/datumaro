# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum, auto


class CocoTask(Enum):
    instances = auto()
    person_keypoints = auto()
    captions = auto()
    labels = auto()  # extension, does not exist in the original COCO format
    image_info = auto()
    panoptic = auto()
    stuff = auto()


class CocoPath:
    IMAGES_DIR = "images"
    ANNOTATIONS_DIR = "annotations"

    IMAGE_EXT = ".jpg"
    PANOPTIC_EXT = ".png"
