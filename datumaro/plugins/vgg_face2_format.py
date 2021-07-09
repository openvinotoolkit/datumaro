# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import csv
import os
import os.path as osp

from datumaro.components.converter import Converter
from datumaro.components.extractor import (
    AnnotationType, Bbox, DatasetItem, Importer, Label, LabelCategories, Points,
    SourceExtractor,
)
from datumaro.util.image import find_images


class VggFace2Path:
    ANNOTATION_DIR = "bb_landmark"
    IMAGE_EXT = '.jpg'
    BBOXES_FILE = 'loose_bb_'
    LANDMARKS_FILE = 'loose_landmark_'
    LABELS_FILE = 'labels.txt'
    IMAGES_DIR_NO_LABEL = 'no_label'

class VggFace2Extractor(SourceExtractor):
    def __init__(self, path, subset=None):
        if not osp.isfile(path):
            raise Exception("Can't read .csv annotation file '%s'" % path)
        self._path = path
        self._dataset_dir = osp.dirname(osp.dirname(path))

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]
            if subset.startswith(VggFace2Path.LANDMARKS_FILE):
                subset = subset.split('_')[2]
        super().__init__(subset=subset)

        self._categories = self._load_categories()
        self._items = list(self._load_items(path).values())

    def _load_categories(self):
        label_cat = LabelCategories()
        path = osp.join(self._dataset_dir, VggFace2Path.LABELS_FILE)
        if osp.isfile(path):
            with open(path, encoding='utf-8') as labels_file:
                lines = [s.strip() for s in labels_file]
            for line in lines:
                objects = line.split()
                label = objects[0]
                class_name = None
                if 1 < len(objects):
                    class_name = objects[1]
                label_cat.add(label, parent=class_name)
        else:
            subset_path = osp.join(self._dataset_dir, self._subset)
            if osp.isdir(subset_path):
                for images_dir in sorted(os.listdir(subset_path)):
                    if osp.isdir(osp.join(subset_path, images_dir)) and \
                            images_dir != VggFace2Path.IMAGES_DIR_NO_LABEL:
                        label_cat.add(images_dir)
        return { AnnotationType.label: label_cat }

    def _load_items(self, path):
        def _get_label(path):
            label_name = path.split('/')[0]
            label = None
            if label_name != VggFace2Path.IMAGES_DIR_NO_LABEL:
                label = \
                    self._categories[AnnotationType.label].find(label_name)[0]
            return label

        items = {}

        image_dir = osp.join(self._dataset_dir, self._subset)
        if osp.isdir(image_dir):
            images = { osp.splitext(osp.relpath(p, image_dir))[0]: p
                for p in find_images(image_dir, recursive=True) }
        else:
            images = {}

        with open(path, encoding='utf-8') as content:
            landmarks_table = list(csv.DictReader(content))
        for row in landmarks_table:
            item_id = row['NAME_ID']
            label = None
            if '/' in item_id:
                label = _get_label(item_id)

            if item_id not in items:
                items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                    image=images.get(row['NAME_ID']))

            annotations = items[item_id].annotations
            if [a for a in annotations if a.type == AnnotationType.points]:
                raise Exception("Item %s: an image can have only one "
                    "set of landmarks" % item_id)

            if len([p for p in row if row[p] == '']) == 0 and len(row) == 11:
                annotations.append(Points(
                    [float(row[p]) for p in row if p != 'NAME_ID'], label=label))
            elif label is not None:
                annotations.append(Label(label=label))

        bboxes_path = osp.join(self._dataset_dir, VggFace2Path.ANNOTATION_DIR,
            VggFace2Path.BBOXES_FILE + self._subset + '.csv')
        if osp.isfile(bboxes_path):
            with open(bboxes_path, encoding='utf-8') as content:
                bboxes_table = list(csv.DictReader(content))
            for row in bboxes_table:
                item_id = row['NAME_ID']
                label = None
                if '/' in item_id:
                    label = _get_label(item_id)

                if item_id not in items:
                    items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                        image=images.get(row['NAME_ID']))

                annotations = items[item_id].annotations
                if [a for a in annotations if a.type == AnnotationType.bbox]:
                    raise Exception("Item %s: an image can have only one "
                        "bbox" % item_id)

                if len([p for p in row if row[p] == '']) == 0 and len(row) == 5:
                    annotations.append(Bbox(float(row['X']), float(row['Y']),
                        float(row['W']), float(row['H']), label=label))
        return items

class VggFace2Importer(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.csv', 'vgg_face2',
            dirname=VggFace2Path.ANNOTATION_DIR,
            file_filter=lambda p: \
                not osp.basename(p).startswith(VggFace2Path.BBOXES_FILE))

class VggFace2Converter(Converter):
    DEFAULT_IMAGE_EXT = VggFace2Path.IMAGE_EXT

    def apply(self):
        save_dir = self._save_dir
        os.makedirs(save_dir, exist_ok=True)

        labels_path = osp.join(save_dir, VggFace2Path.LABELS_FILE)
        labels_file = ''
        for label in self._extractor.categories()[AnnotationType.label]:
            labels_file += '%s' % label.name
            if label.parent:
                labels_file += ' %s' % label.parent
            labels_file += '\n'
        with open(labels_path, 'w', encoding='utf-8') as f:
            f.write(labels_file)

        label_categories = self._extractor.categories()[AnnotationType.label]

        for subset_name, subset in self._extractor.subsets().items():
            bboxes_table = []
            landmarks_table = []
            for item in subset:
                if item.has_image and self._save_images:
                    labels = set(p.label for p in item.annotations
                        if getattr(p, 'label') != None)
                    if labels:
                        for label in labels:
                            self._save_image(item, subdir=osp.join(subset_name,
                                label_categories[label].name))
                    else:
                        self._save_image(item, subdir=osp.join(subset_name,
                            VggFace2Path.IMAGES_DIR_NO_LABEL))

                landmarks = [a for a in item.annotations
                    if a.type == AnnotationType.points]
                if 1 < len(landmarks):
                    raise Exception("Item (%s, %s): an image can have only one "
                        "set of landmarks" % (item.id, item.subset))
                if landmarks:
                    if landmarks[0].label is not None and \
                            label_categories[landmarks[0].label].name:
                        name_id = label_categories[landmarks[0].label].name \
                            + '/' + item.id
                    else:
                        name_id = VggFace2Path.IMAGES_DIR_NO_LABEL \
                            + '/' + item.id
                    points = landmarks[0].points
                    if len(points) != 10:
                        landmarks_table.append({'NAME_ID': name_id})
                    else:
                        landmarks_table.append({'NAME_ID': name_id,
                            'P1X': points[0], 'P1Y': points[1],
                            'P2X': points[2], 'P2Y': points[3],
                            'P3X': points[4], 'P3Y': points[5],
                            'P4X': points[6], 'P4Y': points[7],
                            'P5X': points[8], 'P5Y': points[9]})

                bboxes = [a for a in item.annotations
                    if a.type == AnnotationType.bbox]
                if 1 < len(bboxes):
                    raise Exception("Item (%s, %s): an image can have only one "
                        "bbox" % (item.id, item.subset))
                if bboxes:
                    if bboxes[0].label is not None and \
                            label_categories[bboxes[0].label].name:
                        name_id = label_categories[bboxes[0].label].name \
                            + '/' + item.id
                    else:
                        name_id = VggFace2Path.IMAGES_DIR_NO_LABEL \
                            + '/' + item.id
                    bboxes_table.append({'NAME_ID': name_id, 'X': bboxes[0].x,
                        'Y': bboxes[0].y, 'W': bboxes[0].w, 'H': bboxes[0].h})

                labels = [a for a in item.annotations
                    if a.type == AnnotationType.label]
                for label in labels:
                    if label.label is not None and \
                            label_categories[label.label].name:
                        name_id = label_categories[label.label].name \
                            + '/' + item.id
                    else:
                        name_id = VggFace2Path.IMAGES_DIR_NO_LABEL \
                            + '/' + item.id
                    landmarks_table.append({'NAME_ID': name_id})

                if not landmarks and not bboxes and not labels:
                    landmarks_table.append({'NAME_ID':
                        VggFace2Path.IMAGES_DIR_NO_LABEL + '/' + item.id})

            landmarks_path = osp.join(save_dir, VggFace2Path.ANNOTATION_DIR,
                VggFace2Path.LANDMARKS_FILE + subset_name + '.csv')
            os.makedirs(osp.dirname(landmarks_path), exist_ok=True)
            with open(landmarks_path, 'w', encoding='utf-8', newline='') as file:
                columns = ['NAME_ID', 'P1X', 'P1Y', 'P2X', 'P2Y',
                    'P3X', 'P3Y', 'P4X', 'P4Y', 'P5X', 'P5Y']
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(landmarks_table)

            if bboxes_table:
                bboxes_path = osp.join(save_dir, VggFace2Path.ANNOTATION_DIR,
                    VggFace2Path.BBOXES_FILE + subset_name + '.csv')
                os.makedirs(osp.dirname(bboxes_path), exist_ok=True)
                with open(bboxes_path, 'w', encoding='utf-8', newline='') as file:
                    columns = ['NAME_ID', 'X', 'Y', 'W', 'H']
                    writer = csv.DictWriter(file, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(bboxes_table)
