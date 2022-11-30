# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum, auto


class IcdarTask(Enum):
    word_recognition = auto()
    text_localization = auto()
    text_segmentation = auto()


class IcdarPath:
    IMAGE_EXT = ".png"
    GT_EXT = ".bmp"
    IMAGES_DIR = "images"
