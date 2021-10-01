# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

from datumaro.components.annotation import (
    AnnotationType, Bbox, Label, LabelCategories, Points,
)
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.util.image import find_images


class CelebaPath:
    IMAGES_DIR = 'Img/img_celeba'
    LABELS_FILE = 'identity_CelebA'
    BBOXES_FILE = 'list_bbox_celeba.txt'
    ATTRS_FILE = 'list_attr_celeba.txt'
    LANDMARKS_FILE = 'list_landmarks_celeba.txt'
    SUBSETS_FILE = 'Eval/list_eval_partition.txt'
    SUBSETS = {'0': 'train', '1': 'val', '2': 'test'}

class CelebaExtractor(SourceExtractor):
    def __init__(self, path, subset=None):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        super().__init__()
        self._anno_dir = osp.dirname(path)

        self._categories = { AnnotationType.label: LabelCategories() }
        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        image_dir = osp.join(osp.dirname(self._anno_dir), CelebaPath.IMAGES_DIR)
        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace('\\', '/'): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        label_categories = self._categories[AnnotationType.label]
        with open(path, encoding='utf-8') as f:
            for line in f:
                item_id, item_ann = self.split_annotation(line)
                label_ids = [int(id) for id in item_ann]
                anno = []
                for label in label_ids:
                    if label > len(label_categories):
                        for i in range(len(label_categories), label + 1):
                            label_categories.add('label_%d' % i)
                    anno.append(Label(label))

                items[item_id] = DatasetItem(id=item_id, image=images.get(item_id),
                    annotations=anno)

        bbox_path = osp.join(self._anno_dir, CelebaPath.BBOXES_FILE)
        if osp.isfile(bbox_path):
            with open(bbox_path, encoding='utf-8') as f:
                for line in f:
                    if len(line.split()) < 5 or 'image_id' == line[0:8]:
                        continue
                    item_id, item_ann = self.split_annotation(line)
                    bbox = [float(id) for id in item_ann]
                    if item_id not in items:
                        items[item_id] = DatasetItem(id=item_id, image=images.get(item_id))
                    anno = items[item_id].annotations
                    label = None
                    if 0 < len(anno):
                        label = anno[0].label
                    anno.append(Bbox(bbox[0], bbox[1], bbox[2], bbox[3], label=label))

        attr_path = osp.join(self._anno_dir, CelebaPath.ATTRS_FILE)
        if osp.isfile(attr_path):
            with open(attr_path, encoding='utf-8') as f:
                f.readline()
                line = f.readline()
                name_attr = line.split()
                attr = {}
                for line in f:
                    item_id, item_ann = self.split_annotation(line)
                    if len(name_attr) != len(item_ann):
                        continue
                    for i in range(len(item_ann)):
                            attr[name_attr[i]] = item_ann[i]

                    if item_id not in items:
                        items[item_id] = DatasetItem(id=item_id, image=images.get(item_id))
                    items[item_id].attributes = attr

        landmark_path = osp.join(self._anno_dir, CelebaPath.LANDMARKS_FILE)
        if osp.isfile(landmark_path):
            with open(landmark_path, encoding='utf-8') as f:
                for line in f:
                    if len(line.split()) < 11 or 'lefteye_x' == line[0:9]:
                        continue
                    item_id, item_ann = self.split_annotation(line)
                    landmarks = [float(id) for id in item_ann]
                    if item_id not in items:
                        items[item_id] = DatasetItem(id=item_id, image=images.get(item_id))
                    anno = items[item_id].annotations
                    label = None
                    if 0 < len(anno):
                        label = anno[0].label
                    anno.append(Points(landmarks, label=label))

        subset_path = osp.join(osp.dirname(self._anno_dir), CelebaPath.SUBSETS_FILE)
        if osp.isfile(subset_path):
            with open(subset_path, encoding='utf-8') as f:
                for line in f:
                    item_id, item_ann = self.split_annotation(line)
                    subset = item_ann[0]
                    if item_id not in items:
                        items[item_id] = DatasetItem(id=item_id, image=images.get(item_id))
                    items[item_id].subset = CelebaPath.SUBSETS[subset]
                    if 'default' in self._subsets:
                        self._subsets.pop()
                    if CelebaPath.SUBSETS[subset] not in self._subsets:
                        self._subsets.append(CelebaPath.SUBSETS[subset])

        return items

    def split_annotation(self, line):
        item = line.split('\"')
        if 1 < len(item):
            if len(item) == 3:
                item_id = item[1]
                item = item[2].split()
            else:
                raise Exception("Line %s: unexpected number "
                     "of quotes in filename" % line)
        else:
            item = line.split()
            item_id = osp.splitext(item[0])[0]
        return item_id, item[1:]

class CelebaImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.txt', 'celeba',
            filename=CelebaPath.LABELS_FILE)
