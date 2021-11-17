# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
import json
import os.path as osp

from datumaro.util import find

DATASET_META_FILE = 'dataset_meta.json'

def parse_meta_file(path):
    meta_file = path
    if (osp.isdir(path)):
        meta_file = osp.join(path, DATASET_META_FILE)

    with open(meta_file) as f:
        dataset_meta = json.load(f)

    label_map = OrderedDict()
    colors = dataset_meta.get('segmentation_colors', [])
    parts = dataset_meta.get('parts', {})
    actions = dataset_meta.get('actions', {})

    for i, label in dataset_meta.get('label_map').items():
        label_map[label] = []
        label_map[label].append(None)

        if any(colors) and colors[int(i)] is not None:
            label_map[label][0] = tuple(colors[int(i)])

        label_map[label].append(parts.get(i, []))
        label_map[label].append(actions.get(i, []))

    return label_map

def is_meta_file(path):
    return osp.isfile(osp.join(path, DATASET_META_FILE))

def save_meta_by_label_map(path, format_label_map):
    dataset_meta = {}

    label_map = {}
    segmentation_colors = []
    parts = {}
    actions = {}

    i = 0
    for label_name, label_desc in format_label_map.items():
        label_map[str(i)] = label_name

        segmentation_colors.append([int(label_desc[0][0]), int(label_desc[0][1]), int(label_desc[0][2])]
            if label_desc[0] != None else None)

        if not isinstance(label_desc, tuple) and \
                label_desc[1] is not None and 0 < len(label_desc[1]):
            parts[str(i)] = label_desc[1]

        if not isinstance(label_desc, tuple) and \
                label_desc[2] is not None and 0 < len(label_desc[2]):
            actions[str(i)] = label_desc[2]
        i += 1

    dataset_meta['label_map'] = label_map

    if any(segmentation_colors):
        bg_label = find(format_label_map.items(), lambda x: x[1] == (0, 0, 0))
        if bg_label is not None:
            dataset_meta['background_label'] = str(bg_label[0])

        dataset_meta['segmentation_colors'] = segmentation_colors

    if any(parts):
        dataset_meta['parts'] = parts

    if any(actions):
        dataset_meta['actions'] = actions

    meta_file = osp.join(path, DATASET_META_FILE)

    with open(meta_file, 'w') as f:
        json.dump(dataset_meta, f)
