# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import json
import logging as log
import os
import os.path as osp
import re

import numpy as np

from datumaro.components.extractor import (
    AnnotationType, CompiledMask, DatasetItem, Extractor, Importer,
    LabelCategories, Mask, Polygon,
)
from datumaro.util.image import (
    IMAGE_EXTENSIONS, find_images, lazy_image, load_image,
)


class Ade20k2020Path:
    MASK_PATTERN = re.compile(r'''\w+_seg\.\w+
        | \w+_parts_\d+\.\w+
        | instance_\w+\.\w+
    ''', re.VERBOSE)


class Ade20k2020Extractor(Extractor):
    def __init__(self, path):
        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        subsets = os.listdir(path)
        if len(subsets) < 1:
            raise FileNotFoundError("Can't read subsets in directory '%s'" % path)

        super().__init__(subsets=sorted(subsets))
        self._path = path

        self._items = []
        self._categories  = {}

        for subset in self._subsets:
            self._load_items(subset)

    def __iter__(self):
        return iter(self._items)

    def categories(self):
        return self._categories

    def _load_items(self, subset):
        labels = self._categories.setdefault(AnnotationType.label,
            LabelCategories())
        path = osp.join(self._path, subset)

        images = [i for i in find_images(path, recursive=True)]

        for image_path in sorted(images):
            item_id = osp.splitext(osp.relpath(image_path, path))[0]

            if Ade20k2020Path.MASK_PATTERN.fullmatch(osp.basename(image_path)):
                continue

            item_annotations = []
            item_info = self._load_item_info(image_path)
            for item in item_info:
                label_idx = labels.find(item['label_name'])[0]
                if label_idx is None:
                    labels.add(item['label_name'])

            mask_path = osp.splitext(image_path)[0] + '_seg.png'
            max_part_level = max([p['part_level'] for p in item_info])
            for part_level in range(max_part_level + 1):
                if not osp.exists(mask_path):
                    log.warning('Can`t find part level %s mask for %s' \
                        % (part_level, image_path))
                    continue

                mask = lazy_image(mask_path, loader=self._load_class_mask)
                mask = CompiledMask(instance_mask=mask)

                classes = {(v['class_idx'], v['label_name'])
                    for v in item_info if v['part_level'] == part_level}

                for class_idx, label_name in classes:
                    label_id = labels.find(label_name)[0]
                    item_annotations.append(Mask(label=label_id, id=class_idx,
                        image=mask.lazy_extract(class_idx),
                        group=class_idx, z_order=part_level
                    ))

                mask_path = osp.splitext(image_path)[0] \
                    + '_parts_%s.png' % (part_level + 1)

            for item in item_info:
                instance_path = osp.join(osp.dirname(image_path),
                    item['instance_mask'])
                if not osp.isfile(instance_path):
                    log.warning('Can`t find instance mask: %s' % instance_path)
                    continue

                mask = lazy_image(instance_path, loader=self._load_instance_mask)
                mask = CompiledMask(instance_mask=mask)

                label_id = labels.find(item['label_name'])[0]
                instance_id = item['id']
                attributes = {k: True for k in item['attributes']}
                polygon_points = item['polygon_points']

                item_annotations.append(Mask(label=label_id,
                    image=mask.lazy_extract(1), id=instance_id,
                    attributes=attributes, z_order=item['part_level'],
                    group=instance_id
                ))

                if (len(item['polygon_points']) % 2 == 0 \
                        and 3 <= len(item['polygon_points']) // 2):
                    item_annotations.append(Polygon(polygon_points,
                        label=label_id, attributes=attributes, id=instance_id,
                        z_order=item['part_level'], group=instance_id
                    ))

            self._items.append(DatasetItem(item_id, subset=subset,
                image=image_path, annotations=item_annotations))

    def _load_item_info(self, path):
        json_path = osp.splitext(path)[0] + '.json'
        item_info = []
        if not osp.isfile(json_path):
            raise Exception("Can't find annotation file (*.json) \
                for image %s" % path)

        with open(json_path, 'r', encoding='latin-1') as f:
            item_objects = json.load(f)['annotation']['object']
            for obj in item_objects:
                polygon_points = []
                for x, y in zip(obj['polygon']['x'], obj['polygon']['y']):
                    polygon_points.append(x)
                    polygon_points.append(y)

                attributes = obj['attributes']
                if isinstance(attributes, str):
                    attributes = [attributes]

                item_info.append({
                    'id': obj['id'],
                    'class_idx': obj['name_ndx'],
                    'part_level': obj['parts']['part_level'],
                    'occluded': int(obj['occluded'] == 'yes'),
                    'crop': obj['crop'],
                    'label_name': obj['raw_name'],
                    'attributes': attributes,
                    'instance_mask': obj['instance_mask'],
                    'polygon_points': polygon_points
                })

        return item_info

    @staticmethod
    def _load_instance_mask(path):
        mask = load_image(path)
        _, instance_mask = np.unique(mask, return_inverse=True)
        instance_mask = instance_mask.reshape(mask.shape)
        return instance_mask

    @staticmethod
    def _load_class_mask(path):
        mask = load_image(path)
        mask = ((mask[:, :, 2] / 10).astype(np.int32) << 8) \
            + mask[:, :, 1].astype(np.int32)
        return mask

class Ade20k2020Importer(Importer):
    @classmethod
    def find_sources(cls, path):
        for i in range(5):
            for i in glob.iglob(osp.join(path, *('*' * i))):
                    if osp.splitext(i)[1].lower() in IMAGE_EXTENSIONS:
                        return [{'url': path, 'format': 'ade20k2020'}]
        return []
