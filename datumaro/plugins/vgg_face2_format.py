# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import csv
import os
import os.path as osp

from datumaro.components.converter import Converter
from datumaro.components.extractor import (AnnotationType, Bbox, DatasetItem,
    Importer, Label, LabelCategories, Points, SourceExtractor)


class VggFace2Path:
    ANNOTATION_DIR = "bb_landmark"
    IMAGE_EXT = '.jpg'
    BBOXES_FILE = 'loose_bb_'
    LANDMARKS_FILE = 'loose_landmark_'
    LABELS_FILE = 'labels.txt'
    IMAGES_DIR_NO_LABEL = 'no_label'

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
        items = {}
        with open(path) as content:
            landmarks_table = list(csv.DictReader(content))

        for row in landmarks_table:
            item_id = row['NAME_ID']
            label = None
            if '/' in item_id:
                label_name = item_id.split('/')[0]
                if label_name != VggFace2Path.IMAGES_DIR_NO_LABEL:
                    label = \
                        self._categories[AnnotationType.label].find(label_name)[0]
                item_id = item_id[len(label_name) + 1:]
            if item_id not in items:
                image_path = osp.join(self._dataset_dir, self._subset,
                    row['NAME_ID'] + VggFace2Path.IMAGE_EXT)
                items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                    image=image_path)
            group = 0
            annotations = items[item_id].annotations
            if annotations:
                for anno in annotations:
                    if anno.label == label:
                        raise Exception("A face shouldn't have multiple sets of landmarks")
            if len([p for p in row if row[p] == '']) == 0 and len(row) == 11:
                annotations.append(Points(
                    [float(row[p]) for p in row if p != 'NAME_ID'], label=label,
                    group=group))
            elif label is not None:
                annotations.append(Label(label=label, group=group))

        bboxes_path = osp.join(self._dataset_dir, VggFace2Path.ANNOTATION_DIR,
            VggFace2Path.BBOXES_FILE + self._subset + '.csv')
        if osp.isfile(bboxes_path):
            with open(bboxes_path) as content:
                bboxes_table = list(csv.DictReader(content))
            for row in bboxes_table:
                item_id = row['NAME_ID']
                label = None
                if '/' in item_id:
                    label_name = item_id.split('/')[0]
                    if label_name != VggFace2Path.IMAGES_DIR_NO_LABEL:
                        label = \
                            self._categories[AnnotationType.label].find(label_name)[0]
                    item_id = item_id[len(label_name) + 1:]
                if item_id not in items:
                    image_path = osp.join(self._dataset_dir, self._subset,
                        row['NAME_ID'] + VggFace2Path.IMAGE_EXT)
                    items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                        image=image_path)
                group = 0
                annotations = items[item_id].annotations
                if annotations:
                    max_group = max(annotations, key=lambda x: x.group).group
                    for anno in annotations:
                        if anno.label == label:
                            if anno.group == 0:
                                group = max_group + 1
                                anno.group = group
                            else:
                                group = anno.group
                            break
                if len([p for p in row if row[p] == '']) == 0 and len(row) == 5:
                    annotations.append(Bbox(float(row['X']), float(row['Y']),
                        float(row['W']), float(row['H']), label=label, group=group))
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
            subset_dir = osp.join(save_dir, subset_name)
            bboxes_table = []
            landmarks_table = []
            for item in subset:
                if item.has_image and self._save_images:
                    labels = set(p.label for p in item.annotations
                        if getattr(p, 'label') != None)
                    if labels:
                        for label in labels:
                            self._save_image(item, osp.join(subset_dir,
                                label_categories[label].name + '/' \
                                + item.id + VggFace2Path.IMAGE_EXT))
                    else:
                        self._save_image(item, osp.join(subset_dir,
                            VggFace2Path.IMAGES_DIR_NO_LABEL,
                            item.id + VggFace2Path.IMAGE_EXT))

                landmarks = [a for a in item.annotations
                    if a.type == AnnotationType.points]
                for landmark in landmarks:
                    if landmark.label is not None and \
                            label_categories[landmark.label].name:
                        name_id = label_categories[landmark.label].name \
                            + '/' + item.id
                    else:
                        name_id = VggFace2Path.IMAGES_DIR_NO_LABEL \
                            + '/' + item.id
                    points = landmark.points
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
                for bbox in bboxes:
                    if bbox.label is not None and \
                            label_categories[bbox.label].name:
                        name_id = label_categories[bbox.label].name \
                            + '/' + item.id
                    else:
                        name_id = VggFace2Path.IMAGES_DIR_NO_LABEL \
                            + '/' + item.id
                    bboxes_table.append({'NAME_ID': name_id, 'X': bbox.x,
                        'Y': bbox.y, 'W': bbox.w, 'H': bbox.h})

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
