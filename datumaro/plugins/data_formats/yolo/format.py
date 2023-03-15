# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import IntEnum


class YoloPath:
    DEFAULT_SUBSET_NAME = "train"
    SUBSET_NAMES = ["train", "valid"]
    RESERVED_CONFIG_KEYS = ["backup", "classes", "names"]


class YoloLoosePath:
    NAMES_FILE = "obj.names"


class YoloFormatType(IntEnum):
    yolo_strict = 0
    yolo_loose = 1
    yolo_ultralytics = 2
