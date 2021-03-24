# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from glob import iglob
import os.path as osp

import numpy as np

from datumaro.components.extractor import (Bbox, Caption, DatasetItem,
    Importer, Mask, MaskCategories, Polygon, SourceExtractor)
from datumaro.util.image import find_images
from datumaro.util.mask_tools import lazy_mask

from .format import IcdarPath, IcdarTask


class _IcdarExtractor(SourceExtractor):
    def __init__(self, path, task, subset=None):
        self._path = path
        self._task = task

        if task is IcdarTask.word_recognition:
            if not osp.isfile(path):
                raise FileNotFoundError(
                    "Can't read annotation file '%s'" % path)

            if not subset:
                subset = osp.basename(osp.dirname(path))
            super().__init__(subset=subset)

            self._dataset_dir = osp.dirname(osp.dirname(path))

            self._items = list(self._load_recognition_items().values())
        elif task in {IcdarTask.text_localization, IcdarTask.text_segmentation}:
            if not osp.isdir(path):
                raise NotADirectoryError(
                    "Can't open folder with annotation files '%s'" % path)

            if not subset:
                subset = osp.basename(path)
            super().__init__(subset=subset)

            self._dataset_dir = osp.dirname(path)

            if task is IcdarTask.text_localization:
                self._items = list(self._load_localization_items().values())
            else:
                self._items = list(self._load_segmentation_items().values())

    def _load_recognition_items(self):
        items = {}

        with open(self._path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                objects = line.split(', ')
                if len(objects) == 2:
                    image = objects[0]
                    objects = objects[1].split('\"')
                    if 1 < len(objects):
                        if len(objects) % 2:
                            captions = [objects[2 * i + 1]
                                for i in range(int(len(objects) / 2))]
                        else:
                            raise Exception("Line %s: unexpected number "
                                "of quotes in filename" % line)
                    else:
                        captions = objects[0].split()
                else:
                    image = objects[0][:-1]
                    captions = []

                item_id = osp.splitext(image)[0]
                image_path = osp.join(osp.dirname(self._path),
                    IcdarPath.IMAGES_DIR, image)
                if item_id not in items:
                    items[item_id] = DatasetItem(item_id, subset=self._subset,
                        image=image_path)

                annotations = items[item_id].annotations
                for caption in captions:
                    annotations.append(Caption(caption))

        return items

    def _load_localization_items(self):
        items = {}

        image_dir = osp.join(self._path, IcdarPath.IMAGES_DIR)
        if osp.isdir(image_dir):
            images = { osp.splitext(osp.relpath(p, image_dir))[0]: p
                for p in find_images(image_dir, recursive=True) }
        else:
            images = {}

        for path in iglob(osp.join(self._path, '**', '*.txt'), recursive=True):
            item_id = osp.splitext(osp.relpath(path, self._path))[0]
            if osp.basename(item_id).startswith('gt_'):
                item_id = osp.join(osp.dirname(item_id), osp.basename(item_id)[3:])
            item_id = item_id.replace('\\', '/')

            if item_id not in items:
                items[item_id] = DatasetItem(item_id, subset=self._subset,
                    image=images.get(item_id))
            annotations = items[item_id].annotations

            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    objects = line.split('\"')
                    if 1 < len(objects):
                        if len(objects) == 3:
                            text = objects[1]
                        else:
                            raise Exception("Line %s: unexpected number "
                                "of quotes in filename" % line)
                    else:
                        text = ''
                    objects = objects[0].split()
                    if len(objects) == 1:
                        objects = objects[0].split(',')

                    if 8 <= len(objects):
                        points = [float(p) for p in objects[:8]]

                        attributes = {}
                        if 0 < len(text):
                            attributes['text'] = text
                        elif len(objects) == 9:
                            text = objects[8]
                            attributes['text'] = text

                        annotations.append(
                            Polygon(points, attributes=attributes))
                    elif 4 <= len(objects):
                        x = float(objects[0])
                        y = float(objects[1])
                        w = float(objects[2]) - x
                        h = float(objects[3]) - y

                        attributes = {}
                        if 0 < len(text):
                            attributes['text'] = text
                        elif len(objects) == 5:
                            text = objects[4]
                            attributes['text'] = text

                        annotations.append(
                            Bbox(x, y, w, h, attributes=attributes))
        return items

    def _load_segmentation_items(self):
        items = {}

        image_dir = osp.join(self._path, IcdarPath.IMAGES_DIR)
        if osp.isdir(image_dir):
            images = { osp.splitext(osp.relpath(p, image_dir))[0]: p
                for p in find_images(image_dir, recursive=True) }
        else:
            images = {}

        for path in iglob(osp.join(self._path, '**', '*.txt'), recursive=True):
            item_id = osp.splitext(osp.relpath(path, self._path))[0]
            item_id = item_id.replace('\\', '/')
            if item_id.endswith('_GT'):
                item_id = item_id[:-3]

            if item_id not in items:
                items[item_id] = DatasetItem(item_id, subset=self._subset,
                    image=images.get(item_id))
            annotations = items[item_id].annotations

            colors = [(255, 255, 255)]
            chars = ['']
            centers = [0]
            groups = [0]
            group = 1
            number_in_group = 0
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line == '':
                        if number_in_group == 1:
                            groups[len(groups) - 1] = 0
                        else:
                            group += 1
                        number_in_group = 0
                        continue

                    objects = line.split()
                    if objects[0][0] == '#':
                        objects[0] = objects[0][1:]
                        objects[9] = '\" \"'
                        objects.pop()
                    if len(objects) != 10:
                        raise Exception("Line %s contains the wrong number "
                            "of arguments, e.g. '241 73 144 1 4 0 3 1 4 \"h\"" % line)

                    centers.append(objects[3] + ' ' + objects[4])
                    groups.append(group)
                    colors.append(tuple(int(o) for o in objects[:3]))
                    char = objects[9]
                    if char[0] == '\"' and char[-1] == '\"':
                        char = char[1:-1]
                    chars.append(char)
                    number_in_group += 1
            if number_in_group == 1:
                groups[len(groups) - 1] = 0

            mask_categories = MaskCategories(
                {i: colors[i] for i in range(len(colors))})
            inverse_cls_colormap = mask_categories.inverse_colormap

            gt_path = osp.join(self._path, item_id + '_GT' + IcdarPath.GT_EXT)
            if osp.isfile(gt_path):
                # load mask through cache
                mask = lazy_mask(gt_path, inverse_cls_colormap)
                mask = mask()

                classes = np.unique(mask)
                for label_id in classes:
                    if label_id == 0:
                        continue
                    i = int(label_id)
                    annotations.append(Mask(group=groups[i],
                        image=self._lazy_extract_mask(mask, label_id),
                        attributes={ 'index': i - 1,
                            'color': ' '.join(str(p) for p in colors[i]),
                            'text': chars[i], 'center': centers[i] }
                    ))
        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c

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


class IcdarWordRecognitionImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.txt', 'icdar_word_recognition')

class IcdarTextLocalizationImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '', 'icdar_text_localization')

class IcdarTextSegmentationImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '', 'icdar_text_segmentation')
