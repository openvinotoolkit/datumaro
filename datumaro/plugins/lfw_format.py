# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import re

from datumaro.components.annotation import (
    AnnotationType, Label, LabelCategories, Points,
)
from datumaro.components.converter import Converter
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.util.image import find_images


class LfwPath:
    IMAGES_DIR = 'images'
    LANDMARKS_FILE = 'landmarks.txt'
    PAIRS_FILE = 'pairs.txt'
    PEOPLE_FILE = 'people.txt'
    IMAGE_EXT = '.jpg'
    PATTERN = re.compile(r'([\w-]+)_([-\d]+)')

class LfwExtractor(SourceExtractor):
    def __init__(self, path, subset=None):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        if not subset:
            subset = osp.basename(osp.dirname(path))
        super().__init__(subset=subset)

        self._dataset_dir = osp.dirname(osp.dirname(path))

        people_file = osp.join(osp.dirname(path), LfwPath.PEOPLE_FILE)
        self._categories = self._load_categories(people_file)

        self._items = list(self._load_items(path).values())

    def _load_categories(self, path):
        label_cat = LabelCategories()
        if osp.isfile(path):
            with open(path, encoding='utf-8') as labels_file:
                for line in labels_file:
                    objects = line.strip().split('\t')
                    if len(objects) == 2:
                        label_cat.add(objects[0])
        return { AnnotationType.label: label_cat }

    def _load_items(self, path):
        items = {}
        label_categories = self._categories.get(AnnotationType.label)

        images_dir = osp.join(self._dataset_dir, self._subset, LfwPath.IMAGES_DIR)
        if osp.isdir(images_dir):
            images = { osp.splitext(osp.relpath(p, images_dir))[0].replace('\\', '/'): p
                for p in find_images(images_dir, recursive=True) }
        else:
            images = {}

        with open(path, encoding='utf-8') as f:
            for line in f:
                pair = line.strip().split('\t')
                if len(pair) == 1 and pair[0] != '':
                    annotations = []
                    image = pair[0]
                    item_id = pair[0]
                    objects = item_id.split('/')
                    if 1 < len(objects):
                        label_name = objects[0]
                        label = label_categories.find(label_name)[0]
                        if label != None:
                            annotations.append(Label(label))
                            item_id = item_id[len(label_name) + 1:]
                    if item_id not in items:
                        items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                            image=images.get(image), annotations=annotations)
                elif len(pair) == 3:
                    image1, id1 = self.get_image_name(pair[0], pair[1])
                    image2, id2 = self.get_image_name(pair[0], pair[2])
                    label = label_categories.find(pair[0])[0]
                    if label == None:
                        raise Exception("Line %s: people file doesn't "
                            "contain person %s " % (line, pair[0]))
                    if id1 not in items:
                        annotations = []
                        annotations.append(Label(label))
                        items[id1] = DatasetItem(id=id1, subset=self._subset,
                            image=images.get(image1), annotations=annotations)
                    if id2 not in items:
                        annotations = []
                        annotations.append(Label(label))
                        items[id2] = DatasetItem(id=id2, subset=self._subset,
                            image=images.get(image2), annotations=annotations)

                    # pairs form a directed graph
                    if not items[id1].annotations[0].attributes.get('positive_pairs'):
                        items[id1].annotations[0].attributes['positive_pairs'] = []
                    items[id1].annotations[0].attributes['positive_pairs'].append(image2)

                elif len(pair) == 4:
                    image1, id1 = self.get_image_name(pair[0], pair[1])
                    if pair[2] == '-':
                        image2 = pair[3]
                        id2 = pair[3]
                    else:
                        image2, id2 = self.get_image_name(pair[2], pair[3])
                    if id1 not in items:
                        annotations = []
                        label = label_categories.find(pair[0])[0]
                        if label == None:
                            raise Exception("Line %s: people file doesn't "
                                "contain person %s " % (line, pair[0]))
                        annotations.append(Label(label))
                        items[id1] = DatasetItem(id=id1, subset=self._subset,
                            image=images.get(image1), annotations=annotations)
                    if id2 not in items:
                        annotations = []
                        label = label_categories.find(pair[2])[0]
                        if label != None:
                            annotations.append(Label(label))
                        items[id2] = DatasetItem(id=id2, subset=self._subset,
                            image=images.get(image2), annotations=annotations)

                    # pairs form a directed graph
                    if not items[id1].annotations[0].attributes.get('negative_pairs'):
                        items[id1].annotations[0].attributes['negative_pairs'] = []
                    items[id1].annotations[0].attributes['negative_pairs'].append(image2)

        landmarks_file = osp.join(self._dataset_dir, self._subset,
            LfwPath.LANDMARKS_FILE)
        if osp.isfile(landmarks_file):
            with open(landmarks_file, encoding='utf-8') as f:
                for line in f:
                    line = line.split('\t')

                    item_id = osp.splitext(line[0])[0]
                    objects = item_id.split('/')
                    if 1 < len(objects):
                        label_name = objects[0]
                        label = label_categories.find(label_name)[0]
                        if label != None:
                            item_id = item_id[len(label_name) + 1:]
                    if item_id not in items:
                        items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                            image=osp.join(images_dir, line[0]))

                    annotations = items[item_id].annotations
                    annotations.append(Points([float(p) for p in line[1:]]))

        return items

    @staticmethod
    def get_image_name(person, image_id):
        image, item_id = '', ''
        try:
            image_id = int(image_id)
            image = '{}/{}_{:04d}'.format(person, person, image_id)
            item_id = '{}_{:04d}'.format(person, image_id)
        except ValueError:
            image = '{}/{}'.format(person, image_id)
            item_id = image_id
        return image, item_id

class LfwImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        base, ext = osp.splitext(LfwPath.PAIRS_FILE)
        return cls._find_sources_recursive(path, ext, 'lfw', filename=base)

class LfwConverter(Converter):
    DEFAULT_IMAGE_EXT = LfwPath.IMAGE_EXT

    def apply(self):
        for subset_name, subset in self._extractor.subsets().items():
            label_categories = self._extractor.categories()[AnnotationType.label]
            labels = {}
            for label in label_categories:
                f = label.name
                labels[label.name] = 0

            positive_pairs = []
            negative_pairs = []
            neutral_items = []
            landmarks = []
            included_items = []

            for item in subset:
                anns = [ann for ann in item.annotations
                    if ann.type == AnnotationType.label]
                label, label_name = None, None
                if anns:
                    label = anns[0]
                    label_name = label_categories[anns[0].label].name
                    labels[label_name] += 1

                if self._save_images and item.has_image:
                    subdir=osp.join(subset_name, LfwPath.IMAGES_DIR)
                    if label_name:
                        subdir=osp.join(subdir, label_name)
                    self._save_image(item, subdir=subdir)

                if label != None:
                    person1 = label_name
                    num1 = item.id
                    if num1.startswith(person1):
                        num1 = int(num1.replace(person1, '')[1:])
                    curr_item = person1 + '/' + str(num1)

                    if 'positive_pairs' in label.attributes:
                        if curr_item not in included_items:
                            included_items.append(curr_item)
                        for pair in label.attributes['positive_pairs']:
                            search = LfwPath.PATTERN.search(pair)
                            if search:
                                num2 = search.groups()[1]
                                num2 = int(num2)
                            else:
                                num2 = pair
                                if num2.startswith(person1):
                                    num2 = num2.replace(person1, '')[1:]
                            curr_item = person1 + '/' + str(num2)
                            if curr_item not in included_items:
                                included_items.append(curr_item)
                            positive_pairs.append('%s\t%s\t%s' % (person1, num1, num2))

                    if 'negative_pairs' in label.attributes:
                        if curr_item not in included_items:
                            included_items.append(curr_item)
                        for pair in label.attributes['negative_pairs']:
                            search = LfwPath.PATTERN.search(pair)
                            curr_item = ''
                            if search:
                                person2, num2 = search.groups()
                                num2 = int(num2)
                                curr_item += person2 + '/'
                            else:
                                person2 = '-'
                                num2 = pair
                                objects = pair.split('/')
                                if 1 < len(objects) and objects[0] in labels:
                                    person2 = objects[0]
                                    num2 = pair.replace(person2, '')[1:]
                                    curr_item += person2 + '/'
                            curr_item += str(num2)
                            if curr_item not in included_items:
                                included_items.append(curr_item)
                            negative_pairs.append('%s\t%s\t%s\t%s' % \
                                (person1, num1, person2, num2))

                    if 'positive_pairs' not in label.attributes and \
                            'negative_pairs' not in label.attributes and \
                            curr_item not in included_items:
                        neutral_items.append('%s/%s' % (person1, item.id))
                        included_items.append(curr_item)

                elif item.id not in included_items:
                    neutral_items.append(item.id)
                    included_items.append(item.id)

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
                f.writelines(['%s\n' % item for item in neutral_items])

            if landmarks:
                landmarks_file = osp.join(self._save_dir, subset_name,
                    LfwPath.LANDMARKS_FILE)
                with open(landmarks_file, 'w', encoding='utf-8') as f:
                    f.writelines(['%s\n' % landmark for landmark in landmarks])

            if labels:
                people_file = osp.join(self._save_dir, subset_name,
                    LfwPath.PEOPLE_FILE)
                with open(people_file, 'w', encoding='utf-8') as f:
                    f.writelines(['%s\t%d\n' % (label, labels[label])
                        for label in labels])
