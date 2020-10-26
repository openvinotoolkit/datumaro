
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from glob import glob
import logging as log
import os
import os.path as osp

from datumaro.components.extractor import (DatasetItem, Label,
    LabelCategories, AnnotationType, SourceExtractor, Importer
)
from datumaro.components.converter import Converter


class ImagenetPath:
    IMAGES_EXT = '.jpg'
    IMAGES_DIR_NO_LABEL = 'no_label'


class ImagenetExtractor(SourceExtractor):
    def __init__(self, path, subset=None):
        assert osp.isdir(path), path
        super().__init__(subset=subset)

        self._categories = self._load_categories(path)
        self._items = list(self._load_items(path).values())

    def _load_categories(self, path):
        label_cat = LabelCategories()
        for images_dir in sorted(os.listdir(path)):
            if images_dir != ImagenetPath.IMAGES_DIR_NO_LABEL:
                label_cat.add(images_dir)
        return { AnnotationType.label: label_cat }

    def _load_items(self, path):
        items = {}
        for image_path in glob(osp.join(path, '*', '*')):
            if osp.splitext(image_path)[1] not in ImagenetPath.IMAGES_EXT:
                continue
            label = osp.basename(osp.dirname(image_path))
            image_name = osp.splitext(osp.basename(image_path))[0][len(label) + 1:]
            item = items.get(image_name)
            if item is None:
                item = DatasetItem(id=image_name, subset=self._subset,
                    image=image_path)
            annotations = item.annotations
            if label != ImagenetPath.IMAGES_DIR_NO_LABEL:
                label = self._categories[AnnotationType.label].find(label)[0]
                annotations.append(Label(label=label))
            items[image_name] = item
        return items


class ImagenetImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        if not osp.isdir(path):
            return []
        return [{ 'url': path, 'format': 'imagenet' }]


class ImagenetConverter(Converter):
    DEFAULT_IMAGE_EXT = ImagenetPath.IMAGES_EXT

    def apply(self):
        if 1 < len(self._extractor.subsets()):
            log.warning("ImageNet format supports exporting only a single "
                "subset, subset information will not be used.")

        subset_dir = self._save_dir
        extractor = self._extractor
        labels = {}
        for item in self._extractor:
            image_name = item.id
            labels[image_name] = set(p.label for p in item.annotations)
            for label in labels[image_name]:
                label_name = extractor.categories()[AnnotationType.label][label].name
                self._save_image(item, osp.join(subset_dir, label_name,
                    '%s_%s%s' % \
                    (label_name, image_name, ImagenetPath.IMAGES_EXT)
                ))

            if not labels[image_name]:
                self._save_image(item, osp.join(subset_dir,
                    ImagenetPath.IMAGES_DIR_NO_LABEL,
                    ImagenetPath.IMAGES_DIR_NO_LABEL + '_' +
                    image_name + ImagenetPath.IMAGES_EXT))
