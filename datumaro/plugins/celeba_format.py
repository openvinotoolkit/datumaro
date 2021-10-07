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
    IMAGES_ALIGN_DIR = 'Img/img_align_celeba'
    LABELS_FILE = 'identity_CelebA'
    BBOXES_FILE = 'list_bbox_celeba.txt'
    ATTRS_FILE = 'list_attr_celeba.txt'
    LANDMARKS_FILE = 'list_landmarks_celeba.txt'
    LANDMARKS_ALIGN_FILE = 'list_landmarks_align_celeba.txt'
    SUBSETS_FILE = 'Eval/list_eval_partition.txt'
    SUBSETS = {'0': 'train', '1': 'val', '2': 'test'}
    BBOXES_HEADER = 'image_id x_1 y_1 width height'
    LANDMARKS_HEADER = 'lefteye_x lefteye_y righteye_x righteye_y ' \
        'nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y'

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

        align_dataset = False

        image_dir = osp.join(osp.dirname(self._anno_dir), CelebaPath.IMAGES_DIR)
        if not osp.isdir(image_dir):
            image_dir = osp.join(osp.dirname(self._anno_dir), CelebaPath.IMAGES_ALIGN_DIR)
            align_dataset = True
        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace('\\', '/'): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            align_dataset = False
            images = {}

        label_categories = self._categories[AnnotationType.label]
        with open(path, encoding='utf-8') as f:
            for line in f:
                item_id, item_ann = self.split_annotation(line)
                label_ids = [int(id) for id in item_ann]
                anno = []
                for label in label_ids:
                    while label >= len(label_categories):
                        label_categories.add('class-%d' % len(label_categories))
                    anno.append(Label(label))

                items[item_id] = DatasetItem(id=item_id, image=images.get(item_id),
                    annotations=anno)

        if not align_dataset:
            bbox_path = osp.join(self._anno_dir, CelebaPath.BBOXES_FILE)
            if osp.isfile(bbox_path):
                with open(bbox_path, encoding='utf-8') as f:
                    bboxes_number = int(f.readline().strip())
                    counter = 0
                    if f.readline().strip() != CelebaPath.BBOXES_HEADER:
                        raise Exception("File %s does not match "
                            "the expected format '%s'" % bbox_path,
                            CelebaPath.BBOXES_HEADER)
                    for line in f:
                        item_id, item_ann = self.split_annotation(line)
                        bbox = [float(id) for id in item_ann]
                        if item_id not in items:
                            items[item_id] = DatasetItem(id=item_id,
                                image=images.get(item_id))
                        anno = items[item_id].annotations
                        label = None
                        if 0 < len(anno):
                            label = anno[0].label
                        anno.append(Bbox(bbox[0], bbox[1], bbox[2], bbox[3], label=label))
                        counter += 1
                    if bboxes_number != counter:
                        raise Exception("File %s: the number of bounding "
                            "boxes does not match the specified number "
                            "at the beginning of the file " % bbox_path)

        attr_path = osp.join(self._anno_dir, CelebaPath.ATTRS_FILE)
        if osp.isfile(attr_path):
            with open(attr_path, encoding='utf-8') as f:
                attr_number = int(f.readline().strip())
                counter = 0
                attr_names = f.readline().split()
                attr = {}
                for line in f:
                    item_id, item_ann = self.split_annotation(line)
                    if len(attr_names) != len(item_ann):
                        raise Exception("Line %s: the number of attributes "
                            "in the line does not match the number at the "
                            "beginning of the file " % line)
                    attr = {name: int(ann) > 0 for name, ann in zip(attr_names, item_ann)}

                    if item_id not in items:
                        items[item_id] = DatasetItem(id=item_id, image=images.get(item_id))
                    items[item_id].attributes = attr
                    counter += 1
                if attr_number != counter:
                    raise Exception("File %s: the number of items with "
                        "attributes does not match the specified number "
                        "at the beginning of the file " % attr_path)
        if align_dataset:
            landmark_path = osp.join(self._anno_dir, CelebaPath.LANDMARKS_ALIGN_FILE)
        else:
            landmark_path = osp.join(self._anno_dir, CelebaPath.LANDMARKS_FILE)
        if osp.isfile(landmark_path):
            with open(landmark_path, encoding='utf-8') as f:
                landmarks_number = int(f.readline().strip())
                counter = 0
                if f.readline().strip() != CelebaPath.LANDMARKS_HEADER:
                    raise Exception("File %s does not match "
                        "the expected format '%s'" % bbox_path,
                        CelebaPath.LANDMARKS_HEADER)
                for line in f:
                    item_id, item_ann = self.split_annotation(line)
                    landmarks = [float(id) for id in item_ann]
                    if item_id not in items:
                        items[item_id] = DatasetItem(id=item_id, image=images.get(item_id))
                    anno = items[item_id].annotations
                    label = None
                    if 0 < len(anno):
                        label = anno[0].label
                    anno.append(Points(landmarks, label=label))
                    counter += 1
                if landmarks_number != counter:
                    raise Exception("File %s: the number of landmarks "
                        "does not match the specified number at the "
                        "beginning of the file " % landmark_path)

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
                item_id = osp.splitext(item[1])[0]
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
