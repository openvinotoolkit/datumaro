# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from genericpath import isfile
import os.path as osp
import numpy as np

from attr import attributes
from collections import OrderedDict

from datumaro.components.annotation import (
    AnnotationType, Bbox, Cuboid3d, Label, LabelCategories, MaskCategories, Points, PointsCategories,
)
from datumaro.components.errors import DatasetImportError
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.plugins.camvid_format import parse_label_map
from datumaro.util.image import find_images, load_image
from datumaro.util import find
from datumaro.util.mask_tools import generate_colormap

from glob import glob


class SynthiaPath:
    IMAGES_DIR = 'Img/img_celeba'
    LABELS_FILE = 'Anno/identity_CelebA.txt'
    BBOXES_FILE = 'Anno/list_bbox_celeba.txt'
    ATTRS_FILE = 'Anno/list_attr_celeba.txt'
    LANDMARKS_FILE = 'Anno/list_landmarks_celeba.txt'
    SUBSETS_FILE = 'Eval/list_eval_partition.txt'
    SUBSETS = {'0': 'train', '1': 'val', '2': 'test'}
    BBOXES_HEADER = 'image_id x_1 y_1 width height'

SynthiaLabelMap = {
    0: 'Void',
    1: 'Sky',
    2: 'Building',
    3: 'Road',
    4: 'Sidewalk',
    5: 'Fence',
    6: 'Vegetation',
    7: 'Pole',
    8: 'Car',
    9: 'TrafficSign',
    10: 'Pedestrian',
    11: 'Bicycle',
    12: 'Lanemarking',
    15: 'TrafficLight',
}

SynthiaColorMap = {
    0: (0, 0, 0),
    1: (128, 128, 128),
    2: (128, 0, 0),
    3: (128, 64, 128),
    4: (0, 0, 192),
    5: (64, 64, 128),
    6: (128, 128, 0),
    7: (192, 192, 128),
    8: (64, 0, 128),
    9: (192, 128, 128),
    10: (64, 64, 0),
    11: (0, 128, 192),
    12: (0, 172, 0),
    15: (0, 128, 128),
}

def parse_label_map(path):
    if not path:
        return None

    label_map = OrderedDict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            # skip empty and commented lines
            line = line.strip()
            if not line or line and line[0] == '#':
                continue

            # color, name
            label_desc = line.strip().split()

            if 2 < len(label_desc):
                name = label_desc[3]
                color = tuple([int(c) for c in label_desc[:-1]])
            else:
                name = label_desc[0]
                color = None

            if name in label_map:
                raise ValueError("Label '%s' is already defined" % name)

            label_map[name] = color

    bg_label = find(label_map.items(), lambda x: x[1] == (0, 0, 0))
    if bg_label is not None:
        bg_label = bg_label[0]
    else:
        bg_label = 'background'
        if bg_label not in label_map:
            has_colors = any(v is not None for v in label_map.values())
            color = (0, 0, 0) if has_colors else None
            label_map[bg_label] = color
    label_map.move_to_end(bg_label, last=False)
    return label_map

class SynthiaExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        super().__init__(subset=osp.basename(path))

        self._categories = self._load_categories(path)
        self._items = list(self._load_items(path).values())

    def _load_categories(self, path):
        label_categories = LabelCategories()
        if osp.isfile(osp.join(path, 'labelmap.txt')):
            label_map = parse_label_map(osp.join(path, 'labelmap.txt'))

            for label in label_map:
                label_categories.add(label)

            label_id = lambda label: label_categories.find(label)[0]
            colormap = { label_id(name): (desc[0], desc[1], desc[2])
                for name, desc in label_map.items() if desc }
        else:
            for label in SynthiaLabelMap.values():
                label_categories.add(label)
            colormap = SynthiaColorMap
        mask_categories = MaskCategories(colormap)
        return {AnnotationType.label: label_categories,
            AnnotationType.mask: mask_categories}

    def _load_items(self, root_dir):
        items = {}
        label_categories = self._categories[AnnotationType.label]
        for dir_ in glob(osp.join(root_dir, '**', 'Information'), recursive=True):
            image_dir = osp.join(osp.dirname(dir_), 'RGB')
            if osp.isdir(image_dir):
                images = {
                    osp.join(osp.basename(osp.dirname(osp.dirname(p))),
                    osp.splitext(osp.relpath(p, image_dir))[0].replace('\\', '/')): p
                    for p in find_images(image_dir, recursive=True)
                }
            else:
                images = {}

            for labels_kitti_file in glob(osp.join(osp.dirname(dir_), 'labels_kitti', '**')):
                annotations = []
                group = 1
                with open(labels_kitti_file, encoding='utf-8') as f:
                    for line in f:
                        line = line.split()
                        label = label_categories.find(line[0])[0]
                        if (label == None):
                            label = label_categories.add(line[0])

                        annotations.append(Bbox(float(line[4]), float(line[5]), float(line[6]), float(line[7]),
                            label=label, group=group))
                        annotations.append(Cuboid3d([float(line[8]), float(line[9]), float(line[10])],
                            scale=[float(line[11]), float(line[12]), float(line[13])],
                            rotation=[float(line[14]), float(line[14]), float(line[14])], group=group))

                        group += 1

                        item_id = osp.join(osp.basename(osp.dirname(osp.dirname(labels_kitti_file))),
                            osp.splitext(osp.basename(labels_kitti_file))[0])

                        items[item_id] = DatasetItem(id=item_id, subset=self._subset, image=images.get(item_id),
                            annotations=annotations)

        return items

class SynthiaImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '', 'synthia')