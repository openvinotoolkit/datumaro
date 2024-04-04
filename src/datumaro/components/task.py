# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from enum import IntEnum


class TaskType(IntEnum):
    unknown = 0
    classification = 1
    classification_multilabel = 2
    classification_hierarchical = 3
    detection = 4
    detection_rotated = 5
    detection_3d = 6
    segmentation_semantic = 7
    segmentation_instance = 8
    segmentation_panoptic = 9
    caption = 10
    super_resolution = 11
    depth_estimation = 12
