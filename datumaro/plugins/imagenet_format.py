
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
        labels = {}
        for images_dir in os.listdir(path):
            if images_dir == ImagenetPath.IMAGES_DIR_NO_LABEL:
                for image_path in glob(osp.join(path, images_dir, '*.jpg')):
                    image_name = osp.splitext(osp.basename(image_path))[0]
                    labels[image_name] = []
            else :
                for image_path in glob(osp.join(path, images_dir, '*.jpg')):
                    image_name = osp.splitext(osp.basename(image_path))[0][len(images_dir) + 1:]
                    label_id = self._categories[AnnotationType.label].find(images_dir)[0]
                    if image_name in labels:
                        labels[image_name] += [label_id]
                    else:
                        labels[image_name] = [label_id]

        for image_name in labels:
            if not labels[image_name]:
                image_path = osp.join(path, ImagenetPath.IMAGES_DIR_NO_LABEL,
                    image_name + ImagenetPath.IMAGES_EXT)
                items[image_name] = DatasetItem(
                    id=image_name, image=image_path,
                )
            else:
                image_dir = self._categories[AnnotationType.label][labels[image_name][0]].name
                image_path = osp.join(path, image_dir,
                    image_dir + '_' + image_name + ImagenetPath.IMAGES_EXT)
                items[image_name] = DatasetItem(
                    id=image_name, image=image_path,
                    annotations=[Label(label=p) for p in labels[image_name]]
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
        labels = {}
        for subset in self._extractor.subsets().items():
            for item in subset[1]:
                image_name = item.id
                labels[image_name] =  [p.label for p in item.annotations]
                for label in labels[image_name]:
                    label_name = extractor.categories()[AnnotationType.label][label].name
                    if label not in image_dirs:
                        image_dirs[label] = osp.join(subset_dir, label_name)
                        os.makedirs(image_dirs[label], exist_ok=True)
                    self._save_image(item, osp.join(image_dirs[label],
                        label_name + '_' + image_name + ImagenetPath.IMAGES_EXT))

                if not labels[image_name]:
                    label = -1
                    if label not in image_dirs:
                        image_dirs[label] = osp.join(subset_dir,
                            ImagenetPath.IMAGES_DIR_NO_LABEL)
                        os.makedirs(image_dirs[label], exist_ok=True)
                    self._save_image(item, osp.join(image_dirs[label],
                        image_name + ImagenetPath.IMAGES_EXT))
