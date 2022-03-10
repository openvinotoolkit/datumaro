# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

from datumaro.components.extractor import Importer
from datumaro.components.format_detection import FormatDetectionContext

from .format import VocPath, VocTask


class VocImporter(Importer):
    _TASKS = {
        VocTask.classification: ("voc_classification", "Main"),
        VocTask.detection: ("voc_detection", "Main"),
        VocTask.segmentation: ("voc_segmentation", "Segmentation"),
        VocTask.person_layout: ("voc_layout", "Layout"),
        VocTask.action_classification: ("voc_action", "Action"),
    }

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        # The `voc` format is inherently ambiguous with `voc_classification`,
        # `voc_detection`, etc. To remove the ambiguity (and thus make it
        # possible to use autodetection with the VOC datasets), disable
        # autodetection for the single-task formats.
        if len(cls._TASKS) == 1:
            context.raise_unsupported()

        with context.require_any():
            task_dirs = {task_dir for _, task_dir in cls._TASKS.values()}
            for task_dir in sorted(task_dirs):
                with context.alternative():
                    context.require_file(osp.join(VocPath.SUBSETS_DIR, task_dir, "*.txt"))

    @classmethod
    def find_sources(cls, path):
        subsets = []

        # find root path for the dataset and use it for all tasks
        root_path = None
        for extractor_type, task_dir in cls._TASKS.values():
            if osp.isfile(path) and not osp.basename(osp.dirname(path)) == task_dir:
                continue

            task_subsets = cls._find_sources_recursive(
                root_path or path,
                "txt",
                extractor_type,
                dirname=osp.join(VocPath.SUBSETS_DIR, task_dir),
                file_filter=lambda p: "_" not in osp.basename(p),
                max_depth=0 if root_path else 3,
            )

            if not task_subsets:
                continue

            subsets.extend(task_subsets)

            if not root_path:
                root_path = osp.dirname(osp.dirname(osp.dirname(task_subsets[0]["url"])))

        return subsets


class VocClassificationImporter(VocImporter):
    _TASK = VocTask.classification
    _TASKS = {_TASK: VocImporter._TASKS[_TASK]}


class VocDetectionImporter(VocImporter):
    _TASK = VocTask.detection
    _TASKS = {_TASK: VocImporter._TASKS[_TASK]}


class VocSegmentationImporter(VocImporter):
    _TASK = VocTask.segmentation
    _TASKS = {_TASK: VocImporter._TASKS[_TASK]}


class VocLayoutImporter(VocImporter):
    _TASK = VocTask.person_layout
    _TASKS = {_TASK: VocImporter._TASKS[_TASK]}


class VocActionImporter(VocImporter):
    _TASK = VocTask.action_classification
    _TASKS = {_TASK: VocImporter._TASKS[_TASK]}
