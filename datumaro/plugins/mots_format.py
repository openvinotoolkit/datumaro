# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

# Implements MOTS format https://www.vision.rwth-aachen.de/page/mots

from enum import Enum
from glob import glob
import logging as log
import numpy as np
import os
import os.path as osp

from datumaro.components.extractor import (SourceExtractor, DEFAULT_SUBSET_NAME,
    DatasetItem, AnnotationType, Mask, LabelCategories
)
from datumaro.components.extractor import Importer
from datumaro.components.converter import Converter
from datumaro.util.image import load_image, save_image
from datumaro.util.mask_tools import merge_masks


class MotsPath:
    MASKS_DIR = 'instances'
    IMAGE_DIR = 'images'
    IMAGE_EXT = '.jpg'
    LABELS_FILE = 'labels.txt'
    MAX_INSTANCES = 1000

MotsLabels = Enum('MotsLabels', [
    ('background', 0),
    ('car', 1),
    ('pedestrian', 2),
    ('ignored', 10),
])

class MotsPngExtractor(SourceExtractor):
    @staticmethod
    def detect_dataset(path):
        if osp.isdir(osp.join(path, MotsPath.MASKS_DIR)):
            return [{'url': path, 'format': 'mots_png'}]
        return []

    def __init__(self, path, subset_name=None):
        assert osp.isdir(path), path
        super().__init__(subset=subset_name)
        self._images_dir = osp.join(path, 'images')
        self._anno_dir = osp.join(path, MotsPath.MASKS_DIR)
        self._categories = self._parse_categories(
            osp.join(self._anno_dir, MotsPath.LABELS_FILE))
        self._items = self._parse_items()

    def _parse_categories(self, path):
        if osp.isfile(path):
            with open(path) as f:
                labels = [l.strip() for l in f]
        else:
            labels = [l.name for l in MotsLabels]
        return { AnnotationType.label: LabelCategories.from_iterable(labels) }

    def _parse_items(self):
        items = []
        for p in sorted(p for p in
                glob(self._anno_dir + '/**/*.png', recursive=True)):
            item_id = osp.splitext(osp.relpath(p, self._anno_dir))[0]
            items.append(DatasetItem(id=item_id, subset=self._subset,
                image=osp.join(self._images_dir, item_id + MotsPath.IMAGE_EXT),
                annotations=self._parse_annotations(p)))
        return items

    @staticmethod
    def _lazy_extract_mask(mask, v):
        return lambda: mask == v

    def _parse_annotations(self, path):
        combined_mask = load_image(path, dtype=np.uint16)
        masks = []
        for obj_id in np.unique(combined_mask):
            class_id, instance_id = divmod(obj_id, MotsPath.MAX_INSTANCES)
            z_order = 0
            if class_id == 0:
                continue # background
            if class_id == 10 and \
                    len(self._categories[AnnotationType.label].items) < 10:
                z_order = 1
                class_id = self._categories[AnnotationType.label].find(
                    MotsLabels.ignored.name)[0]
            else:
                class_id -= 1
            masks.append(Mask(self._lazy_extract_mask(combined_mask, obj_id),
                label=class_id, z_order=z_order,
                attributes={'track_id': instance_id}))
        return masks


class MotsImporter(Importer):
    @classmethod
    def find_subsets(cls, path):
        if not osp.isdir(path):
            raise Exception("Expected directory path, got '%s'" % path)
        path = osp.normpath(path)

        subsets = []
        subsets.extend(MotsPngExtractor.detect_dataset(path))
        if not subsets:
            for p in os.listdir(path):
                detected = MotsPngExtractor.detect_dataset(osp.join(path, p))
                for s in detected:
                    s.setdefault('options', {})['subset_name'] = p
                subsets.extend(detected)
        return subsets


class MotsPngConverter(Converter):
    DEFAULT_IMAGE_EXT = MotsPath.IMAGE_EXT

    def apply(self):
        for subset_name in self._extractor.subsets():
            subset = self._extractor.get_subset(subset_name)
            subset_name = subset_name or DEFAULT_SUBSET_NAME

            subset_dir = osp.join(self._save_dir, subset_name)
            images_dir = osp.join(subset_dir, MotsPath.IMAGE_DIR)
            anno_dir = osp.join(subset_dir, MotsPath.MASKS_DIR)

            for item in subset:
                log.debug("Converting item '%s'", item.id)

                if self._save_images:
                    if item.has_image and item.image.has_data:
                        self._save_image(item,
                            osp.join(images_dir, self._make_image_filename(item)))
                    else:
                        log.debug("Item '%s' has no image", item.id)

                self._save_annotations(item, anno_dir)

            with open(osp.join(anno_dir, MotsPath.LABELS_FILE), 'w') as f:
                f.write('\n'.join(l.name
                    for l in subset.categories()[AnnotationType.label].items))

    def _save_annotations(self, item, anno_dir):
        masks = [a for a in item.annotations if a.type == AnnotationType.mask]
        if not masks:
            return

        instance_ids = [int(a.attributes['track_id']) for a in masks]
        masks = sorted(zip(masks, instance_ids), key=lambda e: e[0].z_order)
        mask = merge_masks([
            m.image * (MotsPath.MAX_INSTANCES * (1 + m.label) + id)
            for m, id in masks])
        save_image(osp.join(anno_dir, item.id + '.png'), mask,
            create_dir=True, dtype=np.uint16)
