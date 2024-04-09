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
    detection_landmark = 5
    detection_rotated = 6
    detection_3d = 7
    segmentation_semantic = 8
    segmentation_instance = 9
    segmentation_panoptic = 10
    caption = 11
    super_resolution = 12
    depth_estimation = 13
    mixed = 14
    unlabeled = 15


class TaskAnnotationMapping(Mapping[TaskType, Set[AnnotationType]]):
    def __init__(self):
        self._mapping = {
            TaskType.classification: {AnnotationType.label},
            TaskType.classification_multilabel: {AnnotationType.label},
            TaskType.classification_hierarchical: {AnnotationType.label},
            TaskType.detection: {AnnotationType.label, AnnotationType.bbox},
            TaskType.detection_landmark: {
                AnnotationType.label,
                AnnotationType.bbox,
                AnnotationType.points,
            },
            TaskType.detection_rotated: {
                AnnotationType.label,
                AnnotationType.polygon,
            },
            TaskType.detection_3d: {AnnotationType.label, AnnotationType.cuboid_3d},
            TaskType.segmentation_semantic: {
                AnnotationType.label,
                AnnotationType.mask,
            },
            TaskType.segmentation_instance: {
                AnnotationType.label,
                AnnotationType.bbox,
                AnnotationType.ellipse,
                AnnotationType.polygon,
                AnnotationType.points,
                AnnotationType.polyline,
                AnnotationType.mask,
            },
            TaskType.caption: {AnnotationType.caption},
            TaskType.mixed: {
                AnnotationType.label,
                AnnotationType.bbox,
                AnnotationType.cuboid_3d,
                AnnotationType.ellipse,
                AnnotationType.polygon,
                AnnotationType.points,
                AnnotationType.polyline,
                AnnotationType.mask,
                AnnotationType.caption,
            },
            TaskType.unlabeled: {},
        }

    def __getitem__(self, key):
        return self._mapping[key]

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def get_task(self, ann_types: set[AnnotationType]):
        for task in self._mapping:
            # print(ann_types, task, self._mapping[task])
            if ann_types.issubset(self._mapping[task]):
                return task
        return None
