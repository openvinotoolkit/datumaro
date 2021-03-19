# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum


IcdarTask = Enum('IcdarTask', [
    'word_recognition',
    'text_localization',
    'text_segmentation',
])

class IcdarPath:
    IMAGE_EXT = '.png'
    GT_EXT = '.bmp'
    IMAGES_DIR = 'images'
