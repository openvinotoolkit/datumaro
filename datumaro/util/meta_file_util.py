# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
import json
import os.path as osp

from datumaro.components.annotation import AnnotationType
from datumaro.util import find

DATASET_META_FILE = 'dataset_meta.json'

def is_meta_file(path):
    return osp.splitext(osp.basename(path))[1] == '.json'

def is_meta_file_in_dir(path):
    return osp.isfile(osp.join(path, DATASET_META_FILE))

def parse_meta_file(path):
    meta_file = path
    if (osp.isdir(path)):
        meta_file = osp.join(path, DATASET_META_FILE)

    with open(meta_file) as f:
        dataset_meta = json.load(f)

    label_map = OrderedDict()
    colors = dataset_meta.get('segmentation_colors', [])

    for i, label in dataset_meta.get('label_map').items():
        label_map[label] = None

        if any(colors) and colors[int(i)] is not None:
            label_map[label] = tuple(colors[int(i)])

    return label_map

def save_meta_file(path, categories):
    dataset_meta = {}

    label_map = {}
    for i, label in enumerate(categories[AnnotationType.label]):
        label_map[str(i)] = label.name
    dataset_meta['label_map'] = label_map

    if categories.get(AnnotationType.mask, 0):
        bg_label = find(categories[AnnotationType.mask].colormap.items(),
            lambda x: x[1] == (0, 0, 0))
        if bg_label is not None:
            dataset_meta['background_label'] = str(bg_label[0])

        segmentation_colors = []
        for color in categories[AnnotationType.mask].colormap.values():
            segmentation_colors.append([int(color[0]), int(color[1]), int(color[2])])
        dataset_meta['segmentation_colors'] = segmentation_colors

    meta_file = path
    if osp.isdir(path):
        meta_file = osp.join(path, DATASET_META_FILE)

    with open(meta_file, 'w') as f:
        json.dump(dataset_meta, f)
