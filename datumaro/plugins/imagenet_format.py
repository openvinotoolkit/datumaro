# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp

from datumaro.components.converter import Converter
from datumaro.components.extractor import (
    AnnotationType, DatasetItem, Importer, Label, LabelCategories,
    SourceExtractor,
)
from datumaro.util.image import find_images


class ImagenetPath:
    IMAGE_DIR_NO_LABEL = 'no_label'


class ImagenetExtractor(SourceExtractor):
    def __init__(self, path, subset=None):
        assert osp.isdir(path), path
        super().__init__(subset=subset)

        self._categories = self._load_categories(path)
        self._items = list(self._load_items(path).values())

    def _load_categories(self, path):
        label_cat = LabelCategories()
        for dirname in sorted(os.listdir(path)):
            if dirname != ImagenetPath.IMAGE_DIR_NO_LABEL:
                label_cat.add(dirname)
        return { AnnotationType.label: label_cat }

    def _load_items(self, path):
        items = {}

        for image_path in find_images(path, recursive=True, max_depth=1):
            label = osp.basename(osp.dirname(image_path))
            image_name = osp.splitext(osp.basename(image_path))[0]
            if image_name.startswith(label + '_'):
                image_name = image_name[len(label) + 1:]

            item = items.get(image_name)
            if item is None:
                item = DatasetItem(id=image_name, subset=self._subset,
                    image=image_path)
                items[image_name] = item
            annotations = item.annotations

            if label != ImagenetPath.IMAGE_DIR_NO_LABEL:
                label = self._categories[AnnotationType.label].find(label)[0]
                annotations.append(Label(label=label))

        return items


class ImagenetImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        if not osp.isdir(path):
            return []
        return [{ 'url': path, 'format': 'imagenet' }]


class ImagenetConverter(Converter):
    DEFAULT_IMAGE_EXT = '.jpg'

    def apply(self):
        if 1 < len(self._extractor.subsets()):
            log.warning("ImageNet format only supports exporting a single "
                "subset, subset information will not be used.")

        subset_dir = self._save_dir
        extractor = self._extractor
        labels = {}
        for item in self._extractor:
            labels = set(p.label for p in item.annotations
                if p.type == AnnotationType.label)

            for label in labels:
                label_name = extractor.categories()[AnnotationType.label][label].name
                self._save_image(item, osp.join(subset_dir, label_name,
                    '%s_%s' %  (label_name, self._make_image_filename(item))))

            if not labels:
                self._save_image(item, osp.join(subset_dir,
                    ImagenetPath.IMAGE_DIR_NO_LABEL,
                    ImagenetPath.IMAGE_DIR_NO_LABEL + '_' + \
                    self._make_image_filename(item)))
