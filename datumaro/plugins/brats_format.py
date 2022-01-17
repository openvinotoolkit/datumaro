# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import os.path as osp

import nibabel as nib
import numpy as np

from datumaro.components.annotation import AnnotationType, LabelCategories, Mask
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.components.format_detection import FormatDetectionContext


class BratsPath:
    IMAGES_DIR = 'images'
    LABELS = 'labels'


class BratsExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        self._subset_suffix = osp.basename(path)[len(BratsPath.IMAGES_DIR):]
        subset = None
        if self._subset_suffix == 'Tr':
            subset = 'train'
        elif self._subset_suffix == 'Ts':
            subset = 'test'
        super().__init__(subset=subset)

        self._root_dir = osp.dirname(path)
        self._categories = self._load_categories()
        self._items = list(self._load_items(path).values())

    def _load_categories(self):
        label_cat = LabelCategories()

        labels_path = osp.join(self._root_dir, BratsPath.LABELS)
        if osp.isfile(labels_path):
            with open(labels_path, encoding='utf-8') as f:
                for line in f:
                    label_cat.add(line.strip())

        return { AnnotationType.label: label_cat }

    def _load_items(self, path):
        items = {}

        for image in glob.glob(osp.join(path, '*.nii.gz')):
            data = nib.load(image).get_fdata()
            for i in range(data.shape[2]):
                item_id = '%s_%s' % (osp.splitext(osp.splitext(osp.basename(image))[0])[0], i)

                items[item_id] = DatasetItem(id=item_id, image=data[:,:,i])

        masks_dir = osp.join(self._root_dir, BratsPath.LABELS + self._subset_suffix)
        for mask in glob.glob(osp.join(masks_dir, '*.nii.gz')):
            data = nib.load(mask).get_fdata()
            for i in range(data.shape[2]):
                item_id = '%s_%s' % (osp.splitext(osp.splitext(osp.basename(image))[0])[0], i)

                if item_id not in items:
                    items[item_id] = DatasetItem(id=item_id)

                anno = []
                classes = np.unique(data[:,:,i])
                for class_id in classes:
                    anno.append(Mask(image=self._lazy_extract_mask(data[:,:,i], class_id),
                        label=class_id))

                items[item_id].annotations = anno

        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c

class BratsImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        with context.require_any():
            with context.alternative():
                context.require_file('*/*.nii.gz')

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '', 'brats', filename=f'{BratsPath.IMAGES_DIR}*')
