# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum


IcdarTask = Enum('IcdarTask', [
    'word_recognition',
    'text_localization',
])

class IcdarPath:
    IMAGE_EXT = '.png'
    IMAGES_DIR = 'images'
    VOCABULARY_FILE = 'vocabulary.txt'

    TASK_DIR = {
        IcdarTask.word_recognition: 'word_recognition',
        IcdarTask.text_localization: 'text_localization',
    }
