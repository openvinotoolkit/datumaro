# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum, auto


class DetectionApiPath:
    IMAGES_DIR = "images"
    ANNOTATIONS_DIR = "annotations"

    DEFAULT_IMAGE_EXT = ".jpg"
    IMAGE_EXT_FORMAT = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png"}

    LABELMAP_FILE = "label_map.pbtxt"


class TfrecordImporterType(Enum):
    default = auto()
    roboflow = auto()
