
# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from glob import glob
import os.path as osp

from datumaro.components.extractor import Importer

from .format import VocTask, VocPath


class VocImporter(Importer):
    _TASKS = [
        (VocTask.classification, 'voc_classification', 'Main'),
        (VocTask.detection, 'voc_detection', 'Main'),
        (VocTask.segmentation, 'voc_segmentation', 'Segmentation'),
        (VocTask.person_layout, 'voc_layout', 'Layout'),
        (VocTask.action_classification, 'voc_action', 'Action'),
    ]

    def __call__(self, path, **extra_params):
        subset_paths = self.find_sources(path)
        if len(subset_paths) == 0:
            raise Exception("Failed to find 'voc' dataset at '%s'" % path)

        sources = []
        for _, extractor_type, subset_path in subset_paths:
            sources.append({
                'url': subset_path,
                'format': extractor_type,
                'options': dict(extra_params),
            })

        return sources

    @classmethod
    def find_sources(cls, path):
        subset_paths = []
        for task, extractor_type, task_dir in cls._TASKS:
            task_dir = osp.join(path, VocPath.SUBSETS_DIR, task_dir)
            if not osp.isdir(task_dir):
                continue
            task_subsets = [p for p in glob(osp.join(task_dir, '*.txt'))
                if '_' not in osp.basename(p)]
            subset_paths += [(task, extractor_type, p) for p in task_subsets]
        return subset_paths
