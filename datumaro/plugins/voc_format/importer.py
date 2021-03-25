
# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from glob import glob
import os.path as osp

from datumaro.components.extractor import Importer

from .format import VocTask, VocPath

def find_path(root_path, path, depth=4):
    level, is_found = 0, False
    full_path = None
    while level < depth and not is_found:
        full_path = osp.join(root_path, path)
        paths = glob(full_path)
        if paths:
            full_path = paths[0] # ignore all after the first one
            is_found = osp.isdir(full_path)
        else:
            full_path = None

        level += 1
        root_path = osp.join(root_path, '*')

    return full_path

class VocImporter(Importer):
    _TASKS = [
        ('voc_classification', 'Main'),
        ('voc_detection', 'Main'),
        ('voc_segmentation', 'Segmentation'),
        ('voc_layout', 'Layout'),
        ('voc_action', 'Action'),
    ]

    @classmethod
    def find_sources(cls, path):
        # find root path for the dataset
        root_path = path
        for extractor_type, task_dir in cls._TASKS:
            task_path = find_path(root_path, osp.join(VocPath.SUBSETS_DIR, task_dir))
            if task_path:
                root_path = osp.dirname(osp.dirname(task_path))
                break

        subsets = []
        for extractor_type, task_dir in cls._TASKS:
            task_path = osp.join(root_path, VocPath.SUBSETS_DIR, task_dir)
            if not osp.isdir(task_path):
                continue

            subsets += cls._find_sources_recursive(
                task_path, '.txt', extractor_type, max_depth=0,
                file_filter=lambda p: '_' not in osp.basename(p))
        return subsets
