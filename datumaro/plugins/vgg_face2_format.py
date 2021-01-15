# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import csv
import os
import os.path as osp

from datumaro.components.converter import Converter
from datumaro.components.extractor import (AnnotationType, Bbox, DatasetItem,
    Importer, Points, LabelCategories, SourceExtractor)


class VggFace2Path:
    ANNOTATION_DIR = "bb_landmark"
    IMAGE_EXT = '.jpg'
    BBOXES_FILE = 'loose_bb_'
    LANDMARKS_FILE = 'loose_landmark_'

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

        self._load_categories()
        self._items = list(self._load_items(path).values())

    def _load_categories(self):
        self._categories[AnnotationType.label] = LabelCategories()

    def _load_items(self, path):
        items = {}
        with open(path) as content:
            landmarks_table = list(csv.DictReader(content))

        for row in landmarks_table:
            item_id = row['NAME_ID']
            image_path = osp.join(self._dataset_dir, self._subset,
                item_id + VggFace2Path.IMAGE_EXT)
            annotations = []
            if len([p for p in row if row[p] == '']) == 0 and len(row) == 11:
                annotations.append(Points(
                    [float(row[p]) for p in row if p != 'NAME_ID']))
            if item_id in items and 0 < len(annotations):
                annotation = items[item_id].annotations
                annotation.append(annotations[0])
            else:
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
                    annotations = items[item_id].annotations
                    annotations.append(Bbox(int(row['X']), int(row['Y']),
                        int(row['W']), int(row['H'])))
        return items

class VggFace2Importer(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.csv', 'vgg_face2',
            dirname=VggFace2Path.ANNOTATION_DIR,
            file_filter=lambda p: \
                not osp.basename(p).startswith(VggFace2Path.BBOXES_FILE))

class VggFace2Converter(Converter):
    DEFAULT_IMAGE_EXT = '.jpg'

    def apply(self):
        save_dir = self._save_dir

        os.makedirs(save_dir, exist_ok=True)
        for subset_name, subset in self._extractor.subsets().items():
            subset_dir = osp.join(save_dir, subset_name)
            bboxes_table = []
            landmarks_table = []
            for item in subset:
                if item.has_image and self._save_images:
                    self._save_image(item, osp.join(save_dir, subset_dir,
                        item.id + VggFace2Path.IMAGE_EXT))

                landmarks = [a for a in item.annotations
                    if a.type == AnnotationType.points]
                if landmarks:
                    for landmark in landmarks:
                        points = landmark.points
                        landmarks_table.append({'NAME_ID': item.id,
                            'P1X': points[0], 'P1Y': points[1],
                            'P2X': points[2], 'P2Y': points[3],
                            'P3X': points[4], 'P3Y': points[5],
                            'P4X': points[6], 'P4Y': points[7],
                            'P5X': points[8], 'P5Y': points[9]})
                else:
                    landmarks_table.append({'NAME_ID': item.id})

                bboxes = [a for a in item.annotations
                    if a.type == AnnotationType.bbox]
                if bboxes:
                    for bbox in bboxes:
                        bboxes_table.append({'NAME_ID': item.id, 'X': int(bbox.x),
                            'Y': int(bbox.y), 'W': int(bbox.w), 'H': int(bbox.h)})

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
