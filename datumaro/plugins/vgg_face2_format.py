# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import csv
import os
import os.path as osp
from glob import glob

from datumaro.components.converter import Converter
from datumaro.components.extractor import (AnnotationType, Bbox, DatasetItem,
    Importer, Label, LabelCategories, Points, SourceExtractor)


class VggFace2Path:
    ANNOTATION_DIR = "bb_landmark"
    IMAGE_EXT = '.jpg'
    BBOXES_FILE = 'loose_bb_'
    LANDMARKS_FILE = 'loose_landmark_'
    LABELS_FILE = 'labels.txt'

class VggFace2Extractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isfile(path):
            raise Exception("Can't read .csv annotation file '%s'" % path)
        self._path = path
        self._dataset_dir = osp.dirname(osp.dirname(path))

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
                labels = [s.strip() for s in labels_file]
                for label in labels:
                    label_cat.add(label)
        else:
            subset_path = osp.join(self._dataset_dir, self._subset)
            if osp.isdir(subset_path):
                for images_dir in sorted(os.listdir(subset_path)):
                    if osp.isdir(osp.join(subset_path, images_dir)):
                       label_cat.add(images_dir)
        return { AnnotationType.label: label_cat }

    def _load_items(self, path):
        items = {}
        with open(path) as content:
            landmarks_table = list(csv.DictReader(content))

        for row in landmarks_table:
            item_id = row['NAME_ID']
            label = osp.dirname(item_id)
            if 0 < len(label):
                item_id = osp.basename(item_id)

            points = None
            if len([p for p in row if row[p] == '']) == 0 and len(row) == 11:
                points = Points(
                    [eval(row[p]) for p in row if p != 'NAME_ID'])

            if item_id in items:
                annotation = items[item_id].annotations
                if points is not None:
                    annotation.append(points)
            else:
                image_path = osp.join(self._dataset_dir, self._subset, label,
                    item_id + VggFace2Path.IMAGE_EXT)
                annotations = []
                if 0 < len(label):
                    label = self._categories[AnnotationType.label].find(label)[0]
                    if label != None:
                        annotations.append(Label(label=label))
                if points != None:
                    annotations.append(points)
                items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                    image=image_path, annotations=annotations)

        bboxes_path = osp.join(self._dataset_dir, VggFace2Path.ANNOTATION_DIR,
            VggFace2Path.BBOXES_FILE + self._subset + '.csv')
        if osp.isfile(bboxes_path):
            with open(bboxes_path) as content:
                bboxes_table = list(csv.DictReader(content))
            for row in bboxes_table:
                if len([p for p in row if row[p] == '']) == 0 and len(row) == 5:
                    item_id = row['NAME_ID']
                    label = osp.dirname(item_id)
                    if 0 < len(label):
                        item_id = osp.basename(item_id)

                    annotations = items[item_id].annotations
                    label = self._categories[AnnotationType.label].find(label)[0]
                    if label != None:
                        annotations.append(Bbox(eval(row['X']), eval(row['Y']),
                            eval(row['W']), eval(row['H']), label=label))
                    else:
                        annotations.append(Bbox(eval(row['X']), eval(row['Y']),
                            eval(row['W']), eval(row['H'])))
        return items

class VggFace2Importer(Importer):
    @classmethod
    def find_sources(cls, path):
        subset_paths = [p for p in glob(osp.join(path,
            VggFace2Path.ANNOTATION_DIR, '**.csv'), recursive=True)
            if not osp.basename(p).startswith(VggFace2Path.BBOXES_FILE)]
        sources = []
        for subset_path in subset_paths:
            sources += cls._find_sources_recursive(
                subset_path, '.csv', 'vgg_face2')
        return sources

class VggFace2Converter(Converter):
    DEFAULT_IMAGE_EXT = '.jpg'

    def apply(self):
        save_dir = self._save_dir
        os.makedirs(save_dir, exist_ok=True)

        labels_file = osp.join(save_dir, VggFace2Path.LABELS_FILE)
        with open(labels_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(l.name
                for l in self._extractor.categories()[AnnotationType.label])
            )

        for subset_name, subset in self._extractor.subsets().items():
            subset_dir = osp.join(save_dir, subset_name)
            bboxes_table = []
            landmarks_table = []
            for item in subset:
                labels = [a.label for a in item.annotations
                    if a.type == AnnotationType.label]
                labels_name = [self._extractor.categories()[AnnotationType.label][label].name
                    for label in labels]

                if item.has_image and self._save_images:
                    if 0 < len(labels_name):
                        for label in labels_name:
                            self._save_image(item, osp.join(subset_dir, label,
                                item.id + VggFace2Path.IMAGE_EXT))
                    else:
                        self._save_image(item, osp.join(save_dir, subset_dir,
                            item.id + VggFace2Path.IMAGE_EXT))

                landmarks = [a for a in item.annotations
                    if a.type == AnnotationType.points]
                if labels_name:
                    name_id = labels_name[0] + '/' + item.id
                else:
                    name_id = item.id
                if landmarks:
                    for landmark in landmarks:
                        points = landmark.points
                        landmarks_table.append({'NAME_ID': name_id,
                            'P1X': points[0], 'P1Y': points[1],
                            'P2X': points[2], 'P2Y': points[3],
                            'P3X': points[4], 'P3Y': points[5],
                            'P4X': points[6], 'P4Y': points[7],
                            'P5X': points[8], 'P5Y': points[9]})
                    for i in range(1, len(labels_name) - 1):
                        landmarks_table.append(
                            {'NAME_ID': labels_name[i] + '/' + item.id})
                else:
                    landmarks_table.append({'NAME_ID': name_id})

                bboxes = [a for a in item.annotations
                    if a.type == AnnotationType.bbox]
                for bbox in bboxes:
                    if bbox.label != None:
                        name_id = \
                            self._extractor.categories()[AnnotationType.label][bbox.label].name \
                            + '/' + item.id
                    else:
                        name_id = item.id
                    bboxes_table.append({'NAME_ID': name_id, 'X': bbox.x,
                        'Y': bbox.y, 'W': bbox.w, 'H': bbox.h})

            landmarks_path = osp.join(save_dir, VggFace2Path.ANNOTATION_DIR,
                VggFace2Path.LANDMARKS_FILE + subset_name + '.csv')
            os.makedirs(osp.dirname(landmarks_path), exist_ok=True)
            with open(landmarks_path, 'w', newline='') as file:
                columns = ['NAME_ID', 'P1X', 'P1Y', 'P2X', 'P2Y',
                    'P3X', 'P3Y', 'P4X', 'P4Y', 'P5X', 'P5Y']
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(landmarks_table)

            if bboxes_table:
                bboxes_path = osp.join(save_dir, VggFace2Path.ANNOTATION_DIR,
                    VggFace2Path.BBOXES_FILE + subset_name + '.csv')
                os.makedirs(osp.dirname(bboxes_path), exist_ok=True)
                with open(bboxes_path, 'w', newline='') as file:
                    columns = ['NAME_ID', 'X', 'Y', 'W', 'H']
                    writer = csv.DictWriter(file, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(bboxes_table)
