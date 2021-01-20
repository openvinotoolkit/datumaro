# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from glob import glob

from datumaro.components.extractor import (AnnotationType, Bbox, DatasetItem,
    Importer, Label, LabelCategories, Points, SourceExtractor)
from datumaro.plugins.icdar_format.format import IcdarPath, IcdarTask


class _IcdarExtractor(SourceExtractor):
    def __init__(self, path, task):
        if task is IcdarTask.word_recognition:
            if not osp.isfile(path):
                raise Exception("Can't read annotation file '%s'" % path)
            subset = osp.basename(osp.dirname(path))
            self._dataset_dir = osp.dirname(osp.dirname(osp.dirname(path)))
        elif task is IcdarTask.text_localization:
            if not osp.isdir(path):
                raise Exception("Can't open folder with annotation files'%s'" % path)
            subset = osp.basename(path)
            self._dataset_dir = osp.dirname(osp.dirname(path))
        self._path = path
        self._task = task
        super().__init__(subset=subset)

        self._categories = self._load_categories()
        self._items = list(self._load_items().values())

    def _load_categories(self):
        categories = {}
        path = osp.join(self._dataset_dir, IcdarPath.TASK_DIR[self._task],
            IcdarPath.VOCABULARY_FILE)
        labels = []
        if osp.isfile(path):
            with open(path, encoding='utf-8') as labels_file:
                labels = [s.strip() for s in labels_file]
        else:
            if self._task is IcdarTask.word_recognition:
                paths = [self._path]
            else:
                paths = [p for p in glob(osp.join(self._path, '*.txt'),
                    recursive=True)]
            for path in paths:
                with open(path, encoding='utf-8') as f:
                    for line in f:
                        objects = line.split()
                        if len(objects) == 1:
                            objects = line.split(',')

                        if len(objects) == 9:
                            label = objects[8]
                        elif len(objects) == 5:
                            label = objects[4]
                        elif len(objects) == 2:
                            label = objects[1]
                        else:
                            continue
                        if label[0] == '"' and label[-1] == '"':
                            label = label[1:-1]
                        labels.append(label)

        categories[AnnotationType.label] = \
            LabelCategories().from_iterable(sorted(labels))

        return categories

    def _load_items(self):
        items = {}
        if self._task is IcdarTask.word_recognition:
            with open(self._path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line and line[0] == '#':
                        continue
                    objects = line.split(', ')
                    if len(objects) == 2:
                        image = objects[0]
                        labels = objects[1].split()
                    else:
                        image = objects[0][:-1]
                        labels = []
                    item_id = image[:-len(IcdarPath.IMAGE_EXT)]
                    image_path = osp.join(osp.dirname(self._path),
                        IcdarPath.IMAGES_DIR, image)
                    if item_id not in items:
                        items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                            image=image_path)
                    annotations = items[item_id].annotations
                    for label_name in labels:
                        if label_name[0] == '"' and label_name[-1] == '"':
                            label_name = label_name[1:-1]
                        label = \
                            self._categories[AnnotationType.label]._indices[label_name]
                        annotations.append(Label(label=label))
        elif self._task is IcdarTask.text_localization:
            paths = [p for p in glob(osp.join(self._path, '*.txt'),
                recursive=True)]
            for path in paths:
                item_id = osp.splitext(osp.basename(path))[0]
                if item_id.startswith('gt_'):
                    item_id = item_id[3:]
                image_path = osp.join(self._path, IcdarPath.IMAGES_DIR,
                    item_id + IcdarPath.IMAGE_EXT)
                if item_id not in items:
                    items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                        image=image_path)
                annotations = items[item_id].annotations
                with open(path, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line and line[0] == '#':
                            continue
                        objects = line.split()
                        if len(objects) == 1:
                            objects = line.split(',')
                        if 8 <= len(objects):
                            points = [float(objects[p]) for p in range(8)]
                            label = None
                            if len(objects) == 9:
                                label_name = objects[8]
                                if label_name[0] == '"' and label_name[-1] == '"':
                                    label_name = label_name[1:-1]
                                label = \
                                    self._categories[AnnotationType.label]._indices[label_name]
                            annotations.append(Points(points, label=label))
                        elif 4 <= len(objects):
                            x = float(objects[0])
                            y = float(objects[1])
                            w = float(objects[2]) - x
                            h = float(objects[3]) - y
                            label = None
                            if len(objects) == 5:
                                label_name = objects[4]
                                if label_name[0] == '"' and label_name[-1] == '"':
                                    label_name = label_name[1:-1]
                                label = \
                                    self._categories[AnnotationType.label]._indices[label_name]
                            annotations.append(Bbox(x, y, w, h, label=label))
        return items


class IcdarWordRecognitionExtractor(_IcdarExtractor):
    def __init__(self, path, **kwargs):
        kwargs['task'] = IcdarTask.word_recognition
        super().__init__(path, **kwargs)

class IcdarTextLocalizationExtractor(_IcdarExtractor):
    def __init__(self, path, **kwargs):
        kwargs['task'] = IcdarTask.text_localization
        super().__init__(path, **kwargs)

class IcdarImporter(Importer):
    _TASKS = [
        (IcdarTask.word_recognition, 'icdar_word_recognition', 'word_recognition'),
        (IcdarTask.text_localization, 'icdar_text_localization', 'text_localization'),
    ]

    @classmethod
    def find_sources(cls, path):
        sources = []
        for task, extractor_type, task_dir in cls._TASKS:
            task_path = osp.join(path, task_dir)
            if not osp.isdir(task_path):
                continue
            if task is IcdarTask.word_recognition:
                sources += cls._find_sources_recursive(
                    task_path, '.txt', extractor_type, file_filter=lambda p: \
                    osp.basename(p) != IcdarPath.VOCABULARY_FILE)
            elif task is IcdarTask.text_localization:
                subset_paths = [p for p in glob(osp.join(task_path, '**'))
                    if osp.basename(p) != IcdarPath.VOCABULARY_FILE]
                for subset_path in subset_paths:
                    sources += [{'url': subset_path, 'format': extractor_type}]
        return sources
