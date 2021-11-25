# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
import json
import os.path as osp


DATASET_META_FILE = 'dataset_meta.json'

def is_meta_file(path):
    return osp.splitext(osp.basename(path))[1] == '.json'

def has_meta_file(path):
    return osp.isfile(get_meta_file(path))

def get_meta_file(path):
    return osp.join(path, DATASET_META_FILE)

def parse_meta_file(path):
    meta_file = path
    if osp.isdir(path):
        meta_file = get_meta_file(path)

    with open(meta_file) as f:
        dataset_meta = json.load(f)

    label_map = OrderedDict()
    colors = dataset_meta.get('segmentation_colors', [])

    for i, label in dataset_meta.get('label_map').items():
        label_map[label] = None

        if any(colors) and colors[int(i)] is not None:
            label_map[label] = tuple(colors[int(i)])

    return label_map
