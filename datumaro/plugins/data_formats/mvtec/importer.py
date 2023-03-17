# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from glob import glob

from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer

from .format import MvtecPath, MvtecTask


class MvtecImporter(Importer):
    _TASKS = {
        MvtecTask.classification: "mvtec_classification",
        MvtecTask.detection: "mvtec_detection",
        MvtecTask.segmentation: "mvtec_segmentation",
    }

    @classmethod
    def find_sources(cls, path):
        subset_paths = glob(osp.join(path, "*"), recursive=False)

        # MVTec format should have MvtecPath.MASK_DIR directory.
        if not MvtecPath.MASK_DIR in [osp.basename(path) for path in subset_paths]:
            return []

        sources = []
        for extractor_type in cls._TASKS.values():
            for subset_path in subset_paths:
                if osp.isdir(subset_path) and MvtecPath.MASK_DIR not in subset_path:
                    sources.append(
                        {
                            "url": subset_path,
                            "format": extractor_type,
                            "options": dict({"merge_policy": "union"}),
                        }
                    )

        return sources


class MvtecClassificationImporter(MvtecImporter):
    _TASK = MvtecTask.classification
    _TASKS = {_TASK: MvtecImporter._TASKS[_TASK]}

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        super().detect(context)
        return FormatDetectionConfidence.MEDIUM


class MvtecDetectionImporter(MvtecImporter):
    _TASK = MvtecTask.detection
    _TASKS = {_TASK: MvtecImporter._TASKS[_TASK]}

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        super().detect(context)
        return FormatDetectionConfidence.MEDIUM


class MvtecSegmentationImporter(MvtecImporter):
    _TASK = MvtecTask.segmentation
    _TASKS = {_TASK: MvtecImporter._TASKS[_TASK]}

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        super().detect(context)
        return FormatDetectionConfidence.MEDIUM
