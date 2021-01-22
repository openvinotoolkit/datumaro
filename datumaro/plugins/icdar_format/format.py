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
    VOCABULARY_FILE = 'vocabulary.txt'

    TASK_DIR = {
        IcdarTask.word_recognition: 'word_recognition',
        IcdarTask.text_localization: 'text_localization',
        IcdarTask.text_segmentation: 'text_segmentation',
    }
