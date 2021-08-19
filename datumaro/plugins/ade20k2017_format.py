# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import logging as log
import os
import os.path as osp
import re

import numpy as np

from datumaro.components.extractor import (
    AnnotationType, CompiledMask, DatasetItem, Extractor, Importer,
    LabelCategories, Mask,
)
from datumaro.util.image import (
    IMAGE_EXTENSIONS, find_images, lazy_image, load_image,
)


class Ade20k2017Path:
    MASK_IMAGE_PATTERN = re.compile('_seg|_parts_')

class Ade20k2017Extractor(Extractor):
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

            if Ade20k2017Path.MASK_IMAGE_PATTERN.search(item_id):
                continue

            item_annotations = []

            item_info = self._load_item_info(image_path)
            for item in item_info:
                label_idx = labels.find(item['label_name'])[0]
                if label_idx is None:
                    labels.add(item['label_name'])

            mask_path = osp.splitext(image_path)[0] + '_seg.png'
            if not osp.isfile(mask_path):
                log.warning("Can't find mask for image: %s" % image_path)

            part_level = 0
            max_part_level = max([p['part_level'] for p in item_info])
            for part_level in range(max_part_level + 1):
                if not osp.exists(mask_path):
                    log.warning('Can`t find part level %s mask for %s' \
                        % (part_level, image_path))
                    continue

                mask = lazy_image(mask_path, loader=self._load_instance_mask)
                mask = CompiledMask(instance_mask=mask)

                for v in item_info:
                    if v['part_level'] != part_level:
                        continue

                    label_id = labels.find(v['label_name'])[0]
                    instance_id = v['id']
                    attributes = {k: True for k in v['attributes']}

                    item_annotations.append(Mask(label=label_id,
                        image=mask.lazy_extract(instance_id), id=instance_id,
                        attributes=attributes, z_order=part_level,
                        group=instance_id
                    ))

                mask_path = osp.splitext(image_path)[0] \
                    + '_parts_%s.png' % (part_level + 1)

            self._items.append(DatasetItem(item_id, subset=subset,
                image=image_path, annotations=item_annotations))

    def _load_item_info(self, path):
        attr_path = osp.splitext(path)[0] + '_atr.txt'
        if not osp.isfile(attr_path):
            raise Exception("Can't find annotation file for image %s" % path)

        item_info = []
        with open(attr_path, 'r', encoding='utf-8') as f:
            for line in f:
                columns = [s.strip() for s in line.split('#')]
                if len(columns) != 6:
                    raise Exception('Invalid line in %s' % attr_path)
                if columns[5][0] != '"' or columns[5][-1] != '"':
                    raise Exception('Attributes column are expected \
                        in double quotes, file %s' % attr_path)
                attributes = [s.strip()
                    for s in columns[5][1:-1].split(',') if s]

                item_info.append({
                    'id': int(columns[0]),
                    'part_level': int(columns[1]),
                    'occluded': int(columns[2]),
                    'label_name': columns[4],
                    'attributes': attributes
                })

        return item_info

    @staticmethod
    def _load_instance_mask(path):
        mask = load_image(path)
        _, instance_mask = np.unique(mask[:, :, 0], return_inverse=True)
        instance_mask = instance_mask.reshape(mask[:, :, 0].shape)
        return instance_mask

class Ade20k2017Importer(Importer):
    @classmethod
    def find_sources(cls, path):
        for i in range(5):
            for i in glob.iglob(osp.join(path, *('*' * i))):
                    if osp.splitext(i)[1].lower() in IMAGE_EXTENSIONS:
                        return [{'url': path, 'format': 'ade20k2017'}]
        return []
