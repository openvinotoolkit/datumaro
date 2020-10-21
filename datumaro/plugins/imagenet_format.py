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
    LABELS_FILE = 'synsets.txt'
    IMAGES_DIR = 'data'


class ImagenetExtractor(SourceExtractor):
    def __init__(self, path):
        assert osp.isfile(path), path
        super().__init__(subset=osp.splitext(osp.basename(path))[0])

        labels = osp.join(osp.dirname(path), ImagenetPath.LABELS_FILE)
        labels = self._parse_labels(labels)

        self._categories = self._load_categories(labels)
        self._items = list(self._load_items(path).values())

    @staticmethod
    def _parse_labels(path):
        with open(path, encoding='utf-8') as labels_file:
            return [s.strip() for s in labels_file]

    def _load_categories(self, labels):
        label_cat = LabelCategories()
        for label in labels:
            label_cat.add(label)

        return { AnnotationType.label: label_cat }

    def _load_items(self, path):
        items = {}
        images_path = osp.join(osp.dirname(path), ImagenetPath.IMAGES_DIR)
        with open(path, encoding='utf-8') as f:
            for line in f:
                item = line.split()
                image_name = item[0]
                labels_id = item[1:]
                image_path = osp.join(images_path,
                    image_name[:-(len(image_name.split('_')[-1]) + 1)], image_name)
                anno = []
                for label_id in labels_id:
                    anno += [Label(label=label_id)]
                items[image_name[:-4]] = DatasetItem(
                    id=image_name[:-4], subset=self._subset,
                    image=image_path, annotations=anno
                )
        return items


class ImagenetImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        subset_paths = [p for p in glob(osp.join(path, '*.txt'))
                 if 'synsets' not in osp.basename(p)]
        sources = []
        for subset_path in subset_paths:
            sources += cls._find_sources_recursive(subset_path, '.txt', 'imagenet')
        return sources


class ImagenetConverter(Converter):
    DEFAULT_IMAGE_EXT = '.jpg'

    def apply(self):
        subset_dir = self._save_dir
        extractor = self._extractor
        images_dir = osp.join(subset_dir, ImagenetPath.IMAGES_DIR)
        os.makedirs(images_dir, exist_ok=True)
        self._images_dir = images_dir
        image_dirs = {}
        for subset_name, subset in self._extractor.subsets().items():
            annotation_file = osp.join(subset_dir, '%s.txt' % subset_name)
            labels = {}
            for item in subset:
                image_name = self._make_image_filename(item)
                labels[image_name] =  [p.label for p in item.annotations
                                        if p.type == AnnotationType.label]

                if self._save_images:
                    for label in labels[image_name]:
                        if label not in image_dirs:
                            image_dirs[label] = osp.join(images_dir,
                                        extractor.categories()[AnnotationType.label][label].name)
                            os.makedirs(image_dirs[label], exist_ok=True)

                        self._save_image(item, osp.join(image_dirs[label], image_name))

                    if not labels[image_name]:
                        label = -1
                        if label not in image_dirs:
                            image_dirs[label] = osp.join(images_dir, 'no_label')
                            os.makedirs(image_dirs[label], exist_ok=True)

                        self._save_image(item, osp.join(image_dirs[label], image_name))
            with open(annotation_file, 'w', encoding='utf-8') as f:
                f.writelines(['%s %s\n' % (image_name, ' '.join([str(label)
                            for label in labels[image_name]])) for image_name in labels])

        labels_file = osp.join(subset_dir, ImagenetPath.LABELS_FILE)
        with open(labels_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(l.name
                for l in extractor.categories()[AnnotationType.label])
            )
