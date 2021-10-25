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


class AlignCelebaPath:
    IMAGES_DIR = 'Img/img_align_celeba'
    LABELS_FILE = 'identity_CelebA'
    ATTRS_FILE = 'list_attr_celeba.txt'
    LANDMARKS_FILE = 'list_landmarks_align_celeba.txt'
    SUBSETS_FILE = 'Eval/list_eval_partition.txt'
    SUBSETS = {'0': 'train', '1': 'val', '2': 'test'}
    BBOXES_HEADER = 'image_id x_1 y_1 width height'
    LANDMARKS_HEADER = 'lefteye_x lefteye_y righteye_x righteye_y ' \
        'nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y'

class AlignCelebaExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        super().__init__()
        self._anno_dir = osp.dirname(path)

        self._categories = { AnnotationType.label: LabelCategories() }
        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        image_dir = osp.join(osp.dirname(self._anno_dir),
            AlignCelebaPath.IMAGES_DIR)

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
                    while len(label_categories) <= label:
                        label_categories.add('class-%d' % len(label_categories))
                    anno.append(Label(label))

                items[item_id] = DatasetItem(id=item_id,
                    image=images.get(item_id), annotations=anno)

        landmark_path = osp.join(self._anno_dir, AlignCelebaPath.LANDMARKS_FILE)
        if osp.isfile(landmark_path):
            with open(landmark_path, encoding='utf-8') as f:
                landmarks_number = int(f.readline().strip())

                if f.readline().strip() != AlignCelebaPath.LANDMARKS_HEADER:
                    raise DatasetImportError("File '%s': the header "
                        "does not match the expected format '%s'" % \
                        (landmark_path, AlignCelebaPath.LANDMARKS_HEADER))

                counter = 0
                for counter, line in enumerate(f):
                    item_id, item_ann = self.split_annotation(line)
                    landmarks = [float(id) for id in item_ann]

                    if item_id not in items:
                        raise DatasetImportError("File '%s', line %s: "
                            "for this item are not label in %s " % \
                            (landmark_path, line, AlignCelebaPath.LABELS_FILE))

                    anno = items[item_id].annotations
                    label = anno[0].label
                    anno.append(Points(landmarks, label=label))

                if landmarks_number - 1 != counter:
                    raise DatasetImportError("File '%s': the number of "
                        "landmarks does not match the specified number "
                        "at the beginning of the file " % landmark_path)

        attr_path = osp.join(self._anno_dir, AlignCelebaPath.ATTRS_FILE)
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

                if attr_number - 1 != counter:
                    raise DatasetImportError("File %s: the number of items "
                        "with attributes does not match the specified number "
                        "at the beginning of the file " % attr_path)

        subset_path = osp.join(osp.dirname(self._anno_dir),
            AlignCelebaPath.SUBSETS_FILE)
        if osp.isfile(subset_path):
            with open(subset_path, encoding='utf-8') as f:
                for line in f:
                    item_id, item_ann = self.split_annotation(line)
                    subset_id = item_ann[0]
                    subset = AlignCelebaPath.SUBSETS[subset_id]

                    if item_id not in items:
                        items[item_id] = DatasetItem(id=item_id,
                            image=images.get(item_id))

                    items[item_id].subset = subset

                    if 'default' in self._subsets:
                        self._subsets.pop()
                    self._subsets.append(subset)

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

class AlignCelebaImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.txt', 'align_celeba',
            filename=AlignCelebaPath.LABELS_FILE)
