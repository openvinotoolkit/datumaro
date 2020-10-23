
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
from datumaro.util.image import Image


class ImagenetPath:
    IMAGES_EXT = '.jpg'
    IMAGES_DIR_NO_LABEL = 'no_label'


class ImagenetExtractor(SourceExtractor):
    def __init__(self, path):
        assert osp.isdir(path), path
        super().__init__()

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
            images_dir = osp.basename(osp.dirname(image_path))
            image_name = osp.splitext(osp.basename(image_path))[0][len(images_dir) + 1:]
            item = items.get(image_name)
            if item is None:
                item = DatasetItem(id=image_name, subset=self._subset,
                    image=Image(path=image_path))
            annotations = item.annotations
            if images_dir != ImagenetPath.IMAGES_DIR_NO_LABEL:
                label = self._categories[AnnotationType.label].find(images_dir)[0]
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
        subset_dir = self._save_dir
        extractor = self._extractor
        labels = {}
        for subset in self._extractor.subsets().items():
            for item in subset[1]:
                image_name = item.id
                labels[image_name] =  [p.label for p in item.annotations]
                for label in labels[image_name]:
                    label_name = extractor.categories()[AnnotationType.label][label].name
                    self._save_image(item, osp.join(subset_dir, label_name,
                        label_name + '_' + image_name + ImagenetPath.IMAGES_EXT))

                if not labels[image_name]:
                    self._save_image(item, osp.join(subset_dir,
                        ImagenetPath.IMAGES_DIR_NO_LABEL,
                        ImagenetPath.IMAGES_DIR_NO_LABEL + '_' +
                        image_name + ImagenetPath.IMAGES_EXT))
