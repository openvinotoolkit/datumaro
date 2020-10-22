
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from glob import glob

from datumaro.components.extractor import (DatasetItem, Label,
    LabelCategories, AnnotationType, SourceExtractor, Importer
)
from datumaro.components.converter import Converter


class ImagenetPath:
    IMAGES_EXT = '.jpg'


class ImagenetExtractor(SourceExtractor):
    def __init__(self, path):
        assert osp.isdir(path), path
        super().__init__()

        self._categories = self._load_categories(path)
        self._items = list(self._load_items(path).values())

    def _load_categories(self, path):
        label_cat = LabelCategories()
        for images_dir in os.listdir(path):
            label_cat.add(images_dir)
        return { AnnotationType.label: label_cat }

    def _load_items(self, path):
        items = {}
        for images_dir in os.listdir(path):
            for image_path in glob(osp.join(path, images_dir, '*.jpg')):
                image_name = osp.splitext(osp.basename(image_path))[0]
                label_id = self._categories[AnnotationType.label].find(images_dir)[0]
                items[image_name] = DatasetItem(
                    id=image_name, image=image_path,
                    annotations=[Label(label=label_id)]
                )
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
        subset_dir = self._save_dir
        extractor = self._extractor
        image_dirs = {}
        for subset in self._extractor.subsets().items():
            for item in subset[1]:
                image_name = self._make_image_filename(item)
                if item.annotations[0].type == AnnotationType.label:
                    label = item.annotations[0].label
                    if self._save_images:
                        if label not in image_dirs:
                            image_dirs[label] = osp.join(subset_dir,
                                extractor.categories()[AnnotationType.label][label].name)
                            os.makedirs(image_dirs[label], exist_ok=True)
                        self._save_image(item, osp.join(image_dirs[label], image_name))
