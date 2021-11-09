# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
import os.path as osp

import numpy as np

from datumaro.components.annotation import (
    AnnotationType, LabelCategories, Mask, MaskCategories,
)
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.util import find
from datumaro.util.image import find_images, load_image
from datumaro.util.mask_tools import generate_colormap


class SynthiaPath:
    IMAGES_DIR = 'RGB'
    LABELS_SEGM_DIR = 'GT/LABELS'

SynthiaLabelMap = OrderedDict([
    ('Void', (0, 0, 0)),
    ('Sky', (128, 128, 128)),
    ('Building', (128, 0, 0)),
    ('Road', (128, 64, 128)),
    ('Sidewalk', (0, 0, 192)),
    ('Fence', (64, 64, 128)),
    ('Vegetation', (128, 128, 0)),
    ('Pole', (192, 192, 128)),
    ('Car', (64, 0, 128)),
    ('Truck', (0, 0, 70)),
    ('TrafficSign', (192, 128, 128)),
    ('Pedestrian', (64, 64, 0)),
    ('Bicycle', (0, 128, 192)),
    ('Lanemarking', (0, 172, 0)),
    ('TrafficLight', (0, 128, 128)),
])

def make_categories(label_map=None):
    if label_map is None:
        label_map = SynthiaLabelMap

    # There must always be a label with color (0, 0, 0) at index 0
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

    categories = {}
    label_categories = LabelCategories()
    for label in label_map:
        label_categories.add(label)
    categories[AnnotationType.label] = label_categories

    has_colors = any(v is not None for v in label_map.values())
    if not has_colors: # generate new colors
        colormap = generate_colormap(len(label_map))
    else: # only copy defined colors
        label_id = lambda label: label_categories.find(label)[0]
        colormap = { label_id(name): (desc[0], desc[1], desc[2])
            for name, desc in label_map.items() }
    mask_categories = MaskCategories(colormap)
    mask_categories.inverse_colormap # pylint: disable=pointless-statement
    categories[AnnotationType.mask] = mask_categories
    return categories

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
    return label_map

class SynthiaExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        super().__init__()

        self._categories = self._load_categories(path)
        self._items = list(self._load_items(path).values())

    def _load_categories(self, path):
        label_map = None
        label_map_path = osp.join(path, 'labels.txt')
        if osp.isfile(label_map_path):
            label_map = parse_label_map(label_map_path)
        else:
            label_map = SynthiaLabelMap
        self._labels = [label for label in label_map]
        return make_categories(label_map)

    def _load_items(self, root_dir):
        image_dir = osp.join(root_dir, 'RGB')
        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace('\\', '/'): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        items = {}

        gt_dir = osp.join(root_dir, SynthiaPath.LABELS_SEGM_DIR)
        if osp.isdir(gt_dir):
            gt_images = find_images(gt_dir, recursive=True)
            for gt_img in gt_images:
                item_id = osp.splitext(osp.relpath(gt_img, gt_dir))[0].replace('\\', '/')

                anno = []
                instances_mask = load_image(gt_img, dtype=np.uint16)[:,:,2]
                segm_ids = np.unique(instances_mask)
                for segm_id in segm_ids:
                    anno.append(Mask(
                        image=self._lazy_extract_mask(instances_mask, segm_id),
                        label=segm_id))

                items[item_id] = DatasetItem(id=item_id, image=images[item_id],
                    annotations=anno)
        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c

class SynthiaImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return [{'url': path, 'format': 'synthia'}]
