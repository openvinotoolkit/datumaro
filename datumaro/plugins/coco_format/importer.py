# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from glob import glob
import logging as log
import os.path as osp

from datumaro.components.extractor import DEFAULT_SUBSET_NAME, Importer
from datumaro.components.format_detection import (
    FormatDetectionConfidence, FormatDetectionContext,
)
from datumaro.plugins.coco_format.extractor import (
    CocoCaptionsExtractor, CocoImageInfoExtractor, CocoInstancesExtractor,
    CocoLabelsExtractor, CocoPanopticExtractor, CocoPersonKeypointsExtractor,
    CocoStuffExtractor,
)
from datumaro.util.log_utils import logging_disabled

from .format import CocoTask


class CocoImporter(Importer):
    _TASKS = {
        CocoTask.instances: CocoInstancesExtractor,
        CocoTask.person_keypoints: CocoPersonKeypointsExtractor,
        CocoTask.captions: CocoCaptionsExtractor,
        CocoTask.labels: CocoLabelsExtractor,
        CocoTask.image_info: CocoImageInfoExtractor,
        CocoTask.panoptic: CocoPanopticExtractor,
        CocoTask.stuff: CocoStuffExtractor,
    }

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('--keep-original-category-ids', action='store_true',
            help="Add dummy label categories so that category indices "
                "correspond to the category IDs in the original annotation "
                "file")
        return parser

    @classmethod
    def detect(
        cls, context: FormatDetectionContext,
    ) -> FormatDetectionConfidence:
        with logging_disabled(log.WARN):
            if not cls.find_sources(context.root_path):
                context.fail("specific requirement information unavailable")

        return FormatDetectionConfidence.LOW

    def __call__(self, path, **extra_params):
        subsets = self.find_sources(path)

        if len(subsets) == 0:
            raise Exception("Failed to find 'coco' dataset at '%s'" % path)

        # TODO: should be removed when proper label merging is implemented
        conflicting_types = {CocoTask.instances,
            CocoTask.person_keypoints, CocoTask.labels,
            CocoTask.panoptic, CocoTask.stuff}
        ann_types = set(t for s in subsets.values() for t in s) \
            & conflicting_types
        if 1 <= len(ann_types):
            selected_ann_type = sorted(ann_types, key=lambda x: x.name)[0]
        if 1 < len(ann_types):
            log.warning("Not implemented: "
                "Found potentially conflicting source types with labels: %s. "
                "Only one type will be used: %s" \
                % (", ".join(t.name for t in ann_types), selected_ann_type.name))

        sources = []
        for ann_files in subsets.values():
            for ann_type, ann_file in ann_files.items():
                if ann_type in conflicting_types:
                    if ann_type is not selected_ann_type:
                        log.warning("Not implemented: "
                            "conflicting source '%s' is skipped." % ann_file)
                        continue
                log.info("Found a dataset at '%s'" % ann_file)

                sources.append({
                    'url': ann_file,
                    'format': self._TASKS[ann_type].NAME,
                    'options': dict(extra_params),
                })

        return sources

    @classmethod
    def find_sources(cls, path):
        def detect_coco_task(filename):
            for task in CocoTask:
                if filename.startswith(task.name + '_'):
                    return task
            return None

        if osp.isfile(path):
            if len(cls._TASKS) == 1:
                return {'': { next(iter(cls._TASKS)): path }}

            subset_paths = [path] if path.endswith('.json') else []
        else:
            subset_paths = glob(osp.join(path, '**', '*_*.json'),
                recursive=True)

        subsets = {}
        for subset_path in subset_paths:
            ann_type = detect_coco_task(osp.basename(subset_path))
            if ann_type is None and len(cls._TASKS) == 1:
                ann_type = list(cls._TASKS)[0]

            if ann_type not in cls._TASKS:
                log.warning("File '%s' was skipped, could't match this file "
                    "with any of these tasks: %s" %
                    (subset_path, ','.join(e.NAME for e in cls._TASKS.values()))
                )
                continue

            parts = osp.splitext(osp.basename(subset_path))[0] \
                .split(ann_type.name + '_', maxsplit=1)
            subset_name = parts[1] if len(parts) == 2 else DEFAULT_SUBSET_NAME
            subsets.setdefault(subset_name, {})[ann_type] = subset_path

        return subsets


class CocoImageInfoImporter(CocoImporter):
    _TASK = CocoTask.image_info
    _TASKS = { _TASK: CocoImporter._TASKS[_TASK] }

class CocoCaptionsImporter(CocoImporter):
    _TASK = CocoTask.captions
    _TASKS = { _TASK: CocoImporter._TASKS[_TASK] }

class CocoInstancesImporter(CocoImporter):
    _TASK = CocoTask.instances
    _TASKS = { _TASK: CocoImporter._TASKS[_TASK] }

class CocoPersonKeypointsImporter(CocoImporter):
    _TASK = CocoTask.person_keypoints
    _TASKS = { _TASK: CocoImporter._TASKS[_TASK] }

class CocoLabelsImporter(CocoImporter):
    _TASK = CocoTask.labels
    _TASKS = { _TASK: CocoImporter._TASKS[_TASK] }

class CocoPanopticImporter(CocoImporter):
    _TASK = CocoTask.panoptic
    _TASKS = { _TASK: CocoImporter._TASKS[_TASK] }

class CocoStuffImporter(CocoImporter):
    _TASK = CocoTask.stuff
    _TASKS = { _TASK: CocoImporter._TASKS[_TASK] }
