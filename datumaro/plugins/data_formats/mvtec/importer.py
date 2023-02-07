# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.components.importer import Importer

from .format import MvtecTask


class MvtecImporter(Importer):
    _TASKS = {
        MvtecTask.classification: "mvtec_classification",
        MvtecTask.detection: "mvtec_detection",
        MvtecTask.segmentation: "mvtec_segmentation",
    }

    @classmethod
    def find_sources(cls, path):
        sources = []
        for extractor_type in cls._TASKS.values():
            sources.append(
                {
                    "url": path,
                    "format": extractor_type,
                    "options": dict(),
                }
            )
        return sources


class MvtecClassificationImporter(MvtecImporter):
    _TASK = MvtecTask.classification
    _TASKS = {_TASK: MvtecImporter._TASKS[_TASK]}


class MvtecDetectionImporter(MvtecImporter):
    _TASK = MvtecTask.detection
    _TASKS = {_TASK: MvtecImporter._TASKS[_TASK]}


class MvtecSegmentationImporter(MvtecImporter):
    _TASK = MvtecTask.segmentation
    _TASKS = {_TASK: MvtecImporter._TASKS[_TASK]}
