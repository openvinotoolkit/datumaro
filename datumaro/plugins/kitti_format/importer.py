
# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from glob import glob
import logging as log
import os.path as osp

from datumaro.components.extractor import Importer
from datumaro.util.log_utils import logging_disabled

from .format import KittiTask, KittiPath

class KittiImporter(Importer):
    _TASKS = {
        KittiTask.segmentation: 'kitti_segmentation',
        KittiTask.detection: 'kitti_detection',
    }

    @classmethod
    def detect(cls, path):
        with logging_disabled(log.WARN):
            return len(cls.find_sources(path)) != 0

    def __call__(self, path, **extra_params):
        from datumaro.components.project import Project # cyclic import
        project = Project()

        subsets = self.find_sources(path)

        if len(subsets) == 0:
            raise Exception("Failed to find 'kitti' dataset at '%s'" % path)

        # TODO: should be removed when proper label merging is implemented
        conflicting_types = {KittiTask.segmentation, KittiTask.detection}
        ann_types = set(t for s in subsets.values() for t in s) \
            & conflicting_types
        if 1 <= len(ann_types):
            selected_ann_type = sorted(ann_types, key=lambda x: x.name)[0]
        if 1 < len(ann_types):
            log.warning("Not implemented: "
                "Found potentially conflicting source types with labels: %s. "
                "Only one type will be used: %s" \
                % (", ".join(t.name for t in ann_types), selected_ann_type.name))

        for ann_files in subsets.values():
            for ann_type, ann_file in ann_files.items():
                if ann_type in conflicting_types:
                    if ann_type is not selected_ann_type:
                        log.warning("Not implemented: "
                            "conflicting source '%s' is skipped." % ann_file)
                        continue
                log.info("Found a dataset at '%s'" % ann_file)

                source_name = osp.splitext(osp.basename(ann_file))[0]
                project.add_source(source_name, {
                    'url': ann_file,
                    'format': self._TASKS[ann_type],
                    'options': dict(extra_params),
                })

        return project

    @staticmethod
    def find_sources(path):
        subsets = {}

        for task in KittiPath.TASK_DIR:
            subset_paths = glob(osp.join(path, '**', KittiPath.TASK_DIR[task]),
                recursive=True)
            for subset_path in subset_paths:
                path = osp.normpath(osp.join(subset_path, ".."))
                subset_name = osp.splitext(osp.basename(path))[0]
                subsets.setdefault(subset_name, {})[task] = path

        return subsets

class KittiDetectionImporter(KittiImporter):
    _TASK = KittiTask.detection
    _TASKS = { _TASK: KittiImporter._TASKS[_TASK] }

class KittiSegmentationImporter(KittiImporter):
    _TASK = KittiTask.segmentation
    _TASKS = { _TASK: KittiImporter._TASKS[_TASK] }