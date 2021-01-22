# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from glob import glob

import numpy as np
from datumaro.components.extractor import (AnnotationType, Bbox, Caption,
    DatasetItem, Importer, LabelCategories, Mask, MaskCategories, Polygon,
    SourceExtractor)
from datumaro.util.mask_tools import lazy_mask

from .format import IcdarPath, IcdarTask


class _WordRecognitionExtractor():
    def load_categories(self, _dataset_dir, _path):
        return {}

    def load_items(self, _path, _subset, _categories):
        items = {}
        with open(_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                objects = line.split(', ')
                if len(objects) == 2:
                    image = objects[0]
                    captions = objects[1].split()
                else:
                    image = objects[0][:-1]
                    captions = []
                item_id = image[:-len(IcdarPath.IMAGE_EXT)]
                image_path = osp.join(osp.dirname(_path),
                    IcdarPath.IMAGES_DIR, image)
                if item_id not in items:
                    items[item_id] = DatasetItem(id=item_id, subset=_subset,
                        image=image_path)
                annotations = items[item_id].annotations
                for caption in captions:
                    if caption[0] == '\"' and caption[-1] == '\"':
                        caption = caption[1:-1]
                    annotations.append(Caption(caption))
        return items

class _TextLocalizationExtractor():
    def load_categories(self, _dataset_dir, _path):
        categories = {}
        path = osp.join(_dataset_dir, IcdarPath.VOCABULARY_FILE)
        labels = []
        if osp.isfile(path):
            with open(path, encoding='utf-8') as labels_file:
                labels = [s.strip() for s in labels_file]
        else:
            paths = [p for p in glob(osp.join(_path, '*.txt'))]
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
                        else:
                            continue
                        if label[0] == '\"' and label[-1] == '\"':
                            label = label[1:-1]
                        labels.append(label)

        categories[AnnotationType.label] = \
            LabelCategories().from_iterable(sorted(labels))
        return categories

    def load_items(self, _path, _subset, _categories):
        items = {}
        paths = [p for p in glob(osp.join(_path, '*.txt'))]
        for path in paths:
            item_id = osp.splitext(osp.basename(path))[0]
            if item_id.startswith('gt_'):
                item_id = item_id[3:]
            image_path = osp.join(_path, IcdarPath.IMAGES_DIR,
                item_id + IcdarPath.IMAGE_EXT)
            if item_id not in items:
                items[item_id] = DatasetItem(id=item_id, subset=_subset,
                    image=image_path)
            annotations = items[item_id].annotations
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    objects = line.split()
                    if len(objects) == 1:
                        objects = line.split(',')
                    if 8 <= len(objects):
                        points = [float(objects[p]) for p in range(8)]
                        label = None
                        if len(objects) == 9:
                            label_name = objects[8]
                            if label_name[0] == '\"' and label_name[-1] == '\"':
                                label_name = label_name[1:-1]
                            label = \
                                _categories[AnnotationType.label]._indices[label_name]
                        annotations.append(Polygon(points, label=label))
                    elif 4 <= len(objects):
                        x = float(objects[0])
                        y = float(objects[1])
                        w = float(objects[2]) - x
                        h = float(objects[3]) - y
                        label = None
                        if len(objects) == 5:
                            label_name = objects[4]
                            if label_name[0] == '\"' and label_name[-1] == '\"':
                                label_name = label_name[1:-1]
                            label = \
                                _categories[AnnotationType.label]._indices[label_name]
                        annotations.append(Bbox(x, y, w, h, label=label))
        return items

class _TextSegmentationExtractor():
    def load_categories(self, _dataset_dir, _path):
        return {}

    def load_items(self, _path, _subset, _categories):
        items = {}
        paths = [p for p in glob(osp.join(_path, '*.txt'))]
        for path in paths:
            item_id = osp.splitext(osp.basename(path))[0]
            if item_id.endswith('_GT'):
                item_id = item_id[:-3]
            image_path = osp.join(_path, IcdarPath.IMAGES_DIR,
                item_id + IcdarPath.IMAGE_EXT)
            if item_id not in items:
                items[item_id] = DatasetItem(id=item_id, subset=_subset,
                    image=image_path)
            annotations = items[item_id].annotations
            colors = [(255, 255, 255)]
            chars = ['']
            centers = [0]
            groups = [0]
            group = 0
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line == '':
                        group += 1
                        continue
                    objects = line.split()
                    if objects[0][0] == '#':
                        objects[0] = objects[0][1:]
                        objects[9] = '\" \"'
                        objects.pop()

                    if len(objects) == 10:
                        centers.append([float(objects[3]), float(objects[4])])
                        groups.append(group)
                        colors.append((int(objects[0]), int(objects[1]), int(objects[2])))
                        char = objects[9]
                        if char[0] == '\"' and char[-1] == '\"':
                            char = char[1:-1]
                        chars.append(char)

            mask_categories = MaskCategories({ i: colors[i] for i in range(len(colors)) })
            mask_categories.inverse_colormap # pylint: disable=pointless-statement
            gt_path = osp.join(_path, item_id + '_GT.bmp')
            if osp.isfile(gt_path):
                inverse_cls_colormap = mask_categories.inverse_colormap
                mask = lazy_mask(gt_path, inverse_cls_colormap)
                mask = mask()
                classes = np.unique(mask)
                for label_id in classes:
                    if label_id != 0:
                        image = self._lazy_extract_mask(mask, label_id)
                        i = int(label_id)
                        annotations.append(Mask(image=image, label=i, group=groups[i],
                            attributes={ 'color': colors[i], 'char': chars[i],
                            'center': centers[i] }))

        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c

class _IcdarExtractor(SourceExtractor):
    _TASK_EXTRACTORS = {
        IcdarTask.word_recognition: _WordRecognitionExtractor,
        IcdarTask.text_localization: _TextLocalizationExtractor,
        IcdarTask.text_segmentation: _TextSegmentationExtractor,
    }

    def __init__(self, path, task):
        if task is IcdarTask.word_recognition:
            if not osp.isfile(path):
                raise Exception("Can't read annotation file '%s'" % path)
            subset = osp.basename(osp.dirname(path))
            self._dataset_dir = osp.dirname(osp.dirname(path))
        elif task is IcdarTask.text_localization or \
                task is IcdarTask.text_segmentation:
            if not osp.isdir(path):
                raise Exception("Can't open folder with annotation files'%s'" % path)
            subset = osp.basename(path)
            self._dataset_dir = osp.dirname(path)
        self._path = path
        self._task = task
        super().__init__(subset=subset)

        task_extractor = self._make_task_extractor(task)
        self._categories = task_extractor.load_categories(self._dataset_dir, self._path)
        self._items = list(task_extractor.load_items(self._path, self._subset,
            self._categories).values())

    def _make_task_extractor(self, task):
        if task not in self._TASK_EXTRACTORS:
            raise NotImplementedError()
        return self._TASK_EXTRACTORS[task]()

class IcdarWordRecognitionExtractor(_IcdarExtractor):
    def __init__(self, path, **kwargs):
        kwargs['task'] = IcdarTask.word_recognition
        super().__init__(path, **kwargs)

class IcdarTextLocalizationExtractor(_IcdarExtractor):
    def __init__(self, path, **kwargs):
        kwargs['task'] = IcdarTask.text_localization
        super().__init__(path, **kwargs)

class IcdarTextSegmentationExtractor(_IcdarExtractor):
    def __init__(self, path, **kwargs):
        kwargs['task'] = IcdarTask.text_segmentation
        super().__init__(path, **kwargs)

class IcdarImporter(Importer):
    _TASKS = [
        (IcdarTask.word_recognition, 'icdar_word_recognition', 'word_recognition'),
        (IcdarTask.text_localization, 'icdar_text_localization', 'text_localization'),
        (IcdarTask.text_segmentation, 'icdar_text_segmentation', 'text_segmentation'),
    ]

    @classmethod
    def find_sources(cls, path):
        sources = []
        paths = [path]
        if osp.basename(path) not in IcdarPath.TASK_DIR.values():
            paths = [p for p in glob(osp.join(path, '**'))
                if osp.basename(p) in IcdarPath.TASK_DIR.values()]
        for path in paths:
            for task, extractor_type, task_dir in cls._TASKS:
                if not osp.isdir(path) or osp.basename(path) != task_dir:
                    continue
                if task is IcdarTask.word_recognition:
                    ext = '.txt'
                elif task is IcdarTask.text_localization or \
                        task is IcdarTask.text_segmentation:
                    ext = ''
                sources += cls._find_sources_recursive(
                    path, ext, extractor_type, file_filter=lambda p: \
                    osp.basename(p) != IcdarPath.VOCABULARY_FILE)
            return sources
