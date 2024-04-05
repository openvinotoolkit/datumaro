# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from enum import IntEnum
from typing import Mapping, Set

from datumaro.components.annotation import AnnotationType


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


class TaskAnnotationMapping:
    Mapping[TaskType, Set[AnnotationType]] = {
        TaskType.classification: {AnnotationType.label},
        TaskType.classification_multilabel: {AnnotationType.label},
        TaskType.classification_hierarchical: {AnnotationType.label},
        TaskType.detection: {AnnotationType.label, AnnotationType.bbox},
        TaskType.detection_rotated: {
            AnnotationType.label,
            AnnotationType.polygon,
            AnnotationType.points,
        },
        TaskType.detection_3d: {AnnotationType.label, AnnotationType.cuboid_3d},
        TaskType.segmentation_semantic: {
            AnnotationType.label,
            AnnotationType.polygon,
            AnnotationType.mask,
        },
        TaskType.segmentation_instance: {
            AnnotationType.label,
            AnnotationType.polygon,
            AnnotationType.mask,
        },
        TaskType.segmentation_panoptic: {
            AnnotationType.label,
            AnnotationType.polygon,
            AnnotationType.mask,
        },
    }
