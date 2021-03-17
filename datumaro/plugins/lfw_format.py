# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import re

from datumaro.components.converter import Converter
from datumaro.components.extractor import (AnnotationType, DatasetItem,
    Importer, Points, SourceExtractor)


class LfwPath:
    IMAGES_DIR = 'images'
    LANDMARKS_FILE = 'landmarks.txt'
    PAIRS_FILE = 'pairs.txt'
    IMAGE_EXT = '.jpg'
    PATTERN = re.compile(r'([\w]+)_([-\d]+)')

class LfwExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isfile(path):
            raise NotADirectoryError("Can't read annotation file '%s'" % path)
        super().__init__(subset=osp.basename(osp.dirname(path)))
        self._dataset_dir = osp.dirname(osp.dirname(path))
        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}
        images_dir = osp.join(self._dataset_dir, self._subset, LfwPath.IMAGES_DIR)
        with open(path, encoding='utf-8') as f:
            for line in f:
                pair = line.strip().split('\t')
                if len(pair) == 3:
                    if pair[0] == '-':
                        image1 = pair[1]
                        image2 = pair[2]
                    else:
                        image1 = self.get_image_name(pair[0], pair[1])
                        image2 = self.get_image_name(pair[0], pair[2])
                    if image1 not in items:
                        items[image1] = DatasetItem(id=image1, subset=self._subset,
                            image=osp.join(images_dir, image1 + LfwPath.IMAGE_EXT),
                            attributes={'positive_pairs': [], 'negative_pairs': []})
                    if image2 not in items:
                        items[image2] = DatasetItem(id=image2, subset=self._subset,
                            image=osp.join(images_dir, image2 + LfwPath.IMAGE_EXT),
                            attributes={'positive_pairs': [], 'negative_pairs': []})

                    attributes = items[image1].attributes
                    attributes['positive_pairs'].append(image2)
                elif len(pair) == 4:
                    if pair[0] == '-':
                        image1 = pair[1]
                    else:
                        image1 = self.get_image_name(pair[0], pair[1])
                    if pair[2] == '-':
                        image2 = pair[3]
                    else:
                        image2 = self.get_image_name(pair[2], pair[3])
                    if image1 not in items:
                        items[image1] = DatasetItem(id=image1, subset=self._subset,
                            image=osp.join(images_dir, image1 + LfwPath.IMAGE_EXT),
                            attributes={'positive_pairs': [], 'negative_pairs': []})
                    if image2 not in items:
                        items[image2] = DatasetItem(id=image2, subset=self._subset,
                            image=osp.join(images_dir, image2 + LfwPath.IMAGE_EXT),
                            attributes={'positive_pairs': [], 'negative_pairs': []})

                    attributes = items[image1].attributes
                    attributes['negative_pairs'].append(image2)

        landmarks_file = osp.join(self._dataset_dir, self._subset,
            LfwPath.LANDMARKS_FILE)
        if osp.isfile(landmarks_file):
            with open(landmarks_file, encoding='utf-8') as f:
                for line in f:
                    line = line.split('\t')

                    item_id = line[0]
                    if item_id.endswith(LfwPath.IMAGE_EXT):
                        item_id = item_id[:-len(LfwPath.IMAGE_EXT)]
                    if item_id not in items:
                        items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                            image=osp.join(images_dir, line[0]),
                            attributes={'positive_pairs': [], 'negative_pairs': []})

                    annotations = items[item_id].annotations
                    annotations.append(Points([float(p) for p in line[1:]]))
        return items

    @staticmethod
    def get_image_name(person, image_id):
        return '{}/{}_{:04d}'.format(person, person, int(image_id))

class LfwImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, LfwPath.PAIRS_FILE, 'lfw')

class LfwConverter(Converter):
    DEFAULT_IMAGE_EXT = '.jpg'

    def apply(self):
        for subset_name, subset in self._extractor.subsets().items():
            positive_pairs = []
            negative_pairs = []
            landmarks = []
            for item in subset:
                if item.has_image and self._save_images:
                    self._save_image(item, osp.join(self._save_dir, subset_name,
                        LfwPath.IMAGES_DIR, item.id + LfwPath.IMAGE_EXT))

                search = LfwPath.PATTERN.search(item.id)
                if search:
                    person1, num1 = search.groups()
                    num1 = int(num1)
                else:
                    person1 = '-'
                    num1 = item.id
                if 'positive_pairs' in item.attributes:
                    for pair in item.attributes['positive_pairs']:
                        search = LfwPath.PATTERN.search(pair)
                        if search:
                            num2 = search.groups()[1]
                            num2 = int(num2)
                        else:
                            num2 = pair
                        positive_pairs.append('%s\t%s\t%s' % (person1, num1, num2))
                if 'negative_pairs' in item.attributes:
                    for pair in item.attributes['negative_pairs']:
                        search = LfwPath.PATTERN.search(pair)
                        if search:
                            person2, num2 = search.groups()
                            num2 = int(num2)
                        else:
                            person2 = '-'
                            num2 = pair
                        negative_pairs.append('%s\t%s\t%s\t%s' % \
                            (person1, num1, person2, num2))

                item_landmarks = [p for p in item.annotations
                    if p.type == AnnotationType.points]
                for landmark in item_landmarks:
                    landmarks.append('%s\t%s' % (item.id + LfwPath.IMAGE_EXT,
                        '\t'.join(str(p) for p in landmark.points)))

            pairs_file = osp.join(self._save_dir, subset_name, LfwPath.PAIRS_FILE)
            os.makedirs(osp.dirname(pairs_file), exist_ok=True)
            with open(pairs_file, 'w', encoding='utf-8') as f:
                f.writelines(['%s\n' % pair for pair in positive_pairs])
                f.writelines(['%s\n' % pair for pair in negative_pairs])

            if landmarks:
                landmarks_file = osp.join(self._save_dir, subset_name,
                    LfwPath.LANDMARKS_FILE)
                with open(landmarks_file, 'w', encoding='utf-8') as f:
                    f.writelines(['%s\n' % landmark for landmark in landmarks])
