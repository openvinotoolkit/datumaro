# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

from datumaro.components.annotation import (
    AnnotationType, Bbox, Label, LabelCategories, Points, PointsCategories,
)
from datumaro.components.errors import DatasetImportError
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.util.image import find_images
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class CelebaPath:
    IMAGES_DIR = 'Img/img_celeba'
    LABELS_FILE = 'Anno/identity_CelebA.txt'
    BBOXES_FILE = 'Anno/list_bbox_celeba.txt'
    ATTRS_FILE = 'Anno/list_attr_celeba.txt'
    LANDMARKS_FILE = 'Anno/list_landmarks_celeba.txt'
    SUBSETS_FILE = 'Eval/list_eval_partition.txt'
    SUBSETS = {'0': 'train', '1': 'val', '2': 'test'}
    BBOXES_HEADER = 'image_id x_1 y_1 width height'

class CelebaExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        super().__init__()

        self._categories = { AnnotationType.label: LabelCategories() }
        if has_meta_file(path):
            self._categories = { AnnotationType.label:
                LabelCategories().from_iterable(list(parse_meta_file(path).keys())) }

        self._items = list(self._load_items(path).values())

    def _load_items(self, root_dir):
        items = {}

        image_dir = osp.join(root_dir, CelebaPath.IMAGES_DIR)

        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace('\\', '/'): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        label_categories = self._categories[AnnotationType.label]

        labels_path = osp.join(root_dir, CelebaPath.LABELS_FILE)
        if (not osp.isfile(labels_path)):
            raise DatasetImportError("File '%s': was not found" % labels_path)

        with open(labels_path, encoding='utf-8') as f:
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

        landmark_path = osp.join(root_dir, CelebaPath.LANDMARKS_FILE)
        if osp.isfile(landmark_path):
            with open(landmark_path, encoding='utf-8') as f:
                landmarks_number = int(f.readline().strip())

                point_cat = PointsCategories()
                for i, point_name in enumerate(f.readline().strip().split()):
                    point_cat.add(i, [point_name])
                self._categories[AnnotationType.points] = point_cat

                counter = 0
                for counter, line in enumerate(f):
                    item_id, item_ann = self.split_annotation(line)
                    landmarks = [float(id) for id in item_ann]

                    if len(landmarks) != len(point_cat):
                        raise DatasetImportError("File '%s', line %s: "
                            "points do not match the header of this file" % \
                            (landmark_path, line))

                    if item_id not in items:
                        raise DatasetImportError("File '%s', line %s: "
                            "for this item are not label in %s " % \
                            (landmark_path, line, CelebaPath.LABELS_FILE))

                    anno = items[item_id].annotations
                    label = anno[0].label
                    anno.append(Points(landmarks, label=label))

                if landmarks_number - 1 != counter:
                    raise DatasetImportError("File '%s': the number of "
                        "landmarks does not match the specified number "
                        "at the beginning of the file " % landmark_path)

        bbox_path = osp.join(root_dir, CelebaPath.BBOXES_FILE)
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
                        raise DatasetImportError("File '%s', line %s: "
                            "for this item are not label in %s " % \
                            (bbox_path, line, CelebaPath.LABELS_FILE))

                    anno = items[item_id].annotations
                    label = anno[0].label
                    anno.append(Bbox(
                        bbox[0], bbox[1], bbox[2], bbox[3], label=label))

                if bboxes_number - 1 != counter:
                    raise DatasetImportError(
                        "File '%s': the number of bounding "
                        "boxes does not match the specified number "
                        "at the beginning of the file " % bbox_path)

        attr_path = osp.join(root_dir, CelebaPath.ATTRS_FILE)
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

        subset_path = osp.join(root_dir, CelebaPath.SUBSETS_FILE)
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

class CelebaImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return [{'url': path, 'format': 'celeba'}]
