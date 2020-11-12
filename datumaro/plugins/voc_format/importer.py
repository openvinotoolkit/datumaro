
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
        (VocTask.classification, 'voc_classification', 'Main'),
        (VocTask.detection, 'voc_detection', 'Main'),
        (VocTask.segmentation, 'voc_segmentation', 'Segmentation'),
        (VocTask.person_layout, 'voc_layout', 'Layout'),
        (VocTask.action_classification, 'voc_action', 'Action'),
    ]

    def __call__(self, path, **extra_params):
        from datumaro.components.project import Project # cyclic import
        project = Project()

        subset_paths = self.find_sources(path)
        if len(subset_paths) == 0:
            raise Exception("Failed to find 'voc' dataset at '%s'" % path)

        for task, extractor_type, subset_path in subset_paths:
            project.add_source('%s-%s' %
                (task.name, osp.splitext(osp.basename(subset_path))[0]),
            {
                'url': subset_path,
                'format': extractor_type,
                'options': dict(extra_params),
            })

        return project

    @classmethod
    def find_sources(cls, path):
        subset_paths = []
        for task, extractor_type, task_dir in cls._TASKS:
            task_path = find_path(path, osp.join(VocPath.SUBSETS_DIR, task_dir))

            if not task_path:
                continue
            task_subsets = [p for p in glob(osp.join(task_path, '*.txt'))
                if '_' not in osp.basename(p)]
            subset_paths += [(task, extractor_type, p) for p in task_subsets]
        return subset_paths
