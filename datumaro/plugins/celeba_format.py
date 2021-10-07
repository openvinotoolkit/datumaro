# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

from datumaro.components.annotation import (
    AnnotationType, Bbox, Label, LabelCategories, Points,
)
from datumaro.components.errors import DatasetImportError
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
    def __init__(self, path):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        super().__init__()
        self._anno_dir = osp.dirname(path)

        self._categories = { AnnotationType.label: LabelCategories() }
        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        is_aligned = False

        image_dir = osp.join(osp.dirname(self._anno_dir),
            CelebaPath.IMAGES_DIR)
        if not osp.isdir(image_dir):
            image_dir = osp.join(osp.dirname(self._anno_dir),
                CelebaPath.IMAGES_ALIGN_DIR)
            is_aligned = True

        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace('\\', '/'): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            is_aligned = False
            images = {}

        label_categories = self._categories[AnnotationType.label]
        with open(path, encoding='utf-8') as f:
            for line in f:
                item_id, item_ann = self.split_annotation(line)
                label_ids = [int(id) for id in item_ann]
                anno = []
                for label in label_ids:
                    while len(label_categories) <= label:
                        label_categories.add('class-%d' % len(label_categories))
                    anno.append(Label(label))

                items[item_id] = DatasetItem(id=item_id,
                    image=images.get(item_id), annotations=anno)

        if not is_aligned:
            bbox_path = osp.join(self._anno_dir, CelebaPath.BBOXES_FILE)
            if osp.isfile(bbox_path):
                with open(bbox_path, encoding='utf-8') as f:
                    bboxes_number = int(f.readline().strip())

                    if f.readline().strip() != CelebaPath.BBOXES_HEADER:
                        raise DatasetImportError("File '%s': the header "
                            "does not match the expected format '%s'" % \
                            (bbox_path, CelebaPath.BBOXES_HEADER))

                    counter = 0
                    for counter, line in enumerate(f):
                        item_id, item_ann = self.split_annotation(line)
                        bbox = [float(id) for id in item_ann]

                        if item_id not in items:
                            items[item_id] = DatasetItem(id=item_id,
                                image=images.get(item_id))

                        anno = items[item_id].annotations
                        label = None
                        if anno:
                            label = anno[0].label
                        anno.append(Bbox(
                            bbox[0], bbox[1], bbox[2], bbox[3], label=label))

                    if bboxes_number != counter:
                        raise DatasetImportError(
                            "File '%s': the number of bounding "
                            "boxes does not match the specified number "
                            "at the beginning of the file " % bbox_path)

        attr_path = osp.join(self._anno_dir, CelebaPath.ATTRS_FILE)
        if osp.isfile(attr_path):
            with open(attr_path, encoding='utf-8') as f:
                attr_number = int(f.readline().strip())
                attr_names = f.readline().split()

                counter = 0
                for counter, line in enumerate(f):
                    item_id, item_ann = self.split_annotation(line)
                    if len(attr_names) != len(item_ann):
                        raise DatasetImportError("File '%s', line %s: "
                            "the number of attributes "
                            "in the line does not match the number at the "
                            "beginning of the file " % (attr_path, line))

                    attrs = {name: 0 < int(ann)
                        for name, ann in zip(attr_names, item_ann)}

                    if item_id not in items:
                        items[item_id] = DatasetItem(id=item_id,
                            image=images.get(item_id))

                    items[item_id].attributes = attrs

                if attr_number != counter:
                    raise DatasetImportError("File %s: the number of items "
                        "with attributes does not match the specified number "
                        "at the beginning of the file " % attr_path)

        if is_aligned:
            landmark_path = osp.join(self._anno_dir,
                CelebaPath.LANDMARKS_ALIGN_FILE)
        else:
            landmark_path = osp.join(self._anno_dir,
                CelebaPath.LANDMARKS_FILE)
        if osp.isfile(landmark_path):
            with open(landmark_path, encoding='utf-8') as f:
                landmarks_number = int(f.readline().strip())

                if f.readline().strip() != CelebaPath.LANDMARKS_HEADER:
                    raise DatasetImportError("File '%s': the header "
                        "does not match the expected format '%s'" % \
                        (bbox_path, CelebaPath.LANDMARKS_HEADER))

                counter = 0
                for counter, line in enumerate(f):
                    item_id, item_ann = self.split_annotation(line)
                    landmarks = [float(id) for id in item_ann]

                    if item_id not in items:
                        items[item_id] = DatasetItem(id=item_id,
                            image=images.get(item_id))

                    anno = items[item_id].annotations
                    label = None
                    if anno:
                        label = anno[0].label
                    anno.append(Points(landmarks, label=label))

                if landmarks_number != counter:
                    raise DatasetImportError("File '%s': the number of "
                        "landmarks does not match the specified number "
                        "at the beginning of the file " % landmark_path)

        subset_path = osp.join(osp.dirname(self._anno_dir),
            CelebaPath.SUBSETS_FILE)
        if osp.isfile(subset_path):
            with open(subset_path, encoding='utf-8') as f:
                for line in f:
                    item_id, item_ann = self.split_annotation(line)
                    subset_id = item_ann[0]
                    subset = CelebaPath.SUBSETS[subset_id]

                    if item_id not in items:
                        items[item_id] = DatasetItem(id=item_id,
                            image=images.get(item_id))

                    items[item_id].subset = subset

                    if 'default' in self._subsets:
                        self._subsets.pop()
                    self._subsets.add(subset)

        return items

    def split_annotation(self, line):
        item = line.split('\"')
        if 1 < len(item):
            if len(item) == 3:
                item_id = osp.splitext(item[1])[0]
                item = item[2].split()
            else:
                raise DatasetImportError("Line %s: unexpected number "
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
