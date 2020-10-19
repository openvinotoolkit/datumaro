# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
import os
import os.path as osp
from glob import glob
import logging as log

from datumaro.components.extractor import (DatasetItem, Label,
    LabelCategories, AnnotationType, SourceExtractor, Importer
)
from datumaro.util.log_utils import logging_disabled
from datumaro.components.converter import Converter


class ImagenetPath:
    LABELS_FILE = 'synsets.txt'
    IMAGES_DIR = 'data'


class ImagenetExtractor(SourceExtractor):
    def __init__(self, path):
        assert osp.isfile(path), path
        self._path = path
        self._dataset_dir = osp.dirname(path)

        super().__init__(subset=osp.splitext(osp.basename(path))[0])

        labels = osp.join(osp.dirname(path), ImagenetPath.LABELS_FILE)
        if isinstance(labels, str):
            labels = self._parse_labels(labels)
        elif isinstance(labels, list):
            assert all(isinstance(lbl, str) for lbl in labels), labels
        else:
            raise TypeError("Unexpected type of 'labels' argument: %s" % labels)

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
        items = defaultdict(dict)

        with open(path, encoding='utf-8') as f:
            for line in f:
                image_name, label_id = line.split()
                items[image_name[:-4]] = DatasetItem(
                    id=image_name[:-4], subset=self._subset,
                    image=image_name,
                    annotations=[Label(label=label_id)])
        return items


class ImagenetImporter(Importer):

    @classmethod
    def detect(cls, path):
        with logging_disabled(log.WARN):
            return (len(cls.find_sources(path)) != 0)

    def __call__(self, path, **extra_params):
        from datumaro.components.project import Project # cyclic import
        project = Project()

        subsets = self.find_sources(path)

        if len(subsets) == 0:
            raise Exception("Failed to find 'imagenet' dataset at '%s'" % path)

        for ann_name, ann_file in subsets.items():
            project.add_source(ann_name, {
                    'url': ann_file,
                    'format': 'imagenet',
                    'options': dict(extra_params),
                })

        return project

    @staticmethod
    def find_sources(path):
        subsets = defaultdict(dict)
        subset_paths = [p for p in glob(osp.join(path, '*.txt'))
                if 'synsets' not in osp.basename(p)]

        for subset_path in subset_paths:
            subset_name = osp.splitext(osp.basename(subset_path))[0]
            subsets[subset_name] = subset_path
        return subsets


class ImagenetConverter(Converter):
    DEFAULT_IMAGE_EXT = '.jpg'

    def apply(self):
        subset_dir = self._save_dir
        extractor = self._extractor
        images_dir = osp.join(subset_dir, ImagenetPath.IMAGES_DIR)
        os.makedirs(images_dir, exist_ok=True)
        self._images_dir = images_dir
        image_labels = defaultdict(dict)
        for subset_name, subset in self._extractor.subsets().items():
            annotation_file = osp.join(subset_dir, '%s.txt' % subset_name)
            annotation = ''
            for item in subset:
                label = item.annotations[0].label
                if label not in image_labels:
                    image_dir = osp.join(images_dir, item.id[:-(len(item.id.split('_')[-1]) + 1)])
                    os.makedirs(image_dir, exist_ok=True)
                    image_labels[label] = image_dir
                if not item.has_image:
                    raise Exception("Failed to export item '%s': "
                        "item has no image info" % item.id)
                image_name = self._make_image_filename(item)
                if self._save_images:
                    if item.has_image and item.image.has_data:
                        self._save_image(item, osp.join(image_labels[label], image_name))
                    else:
                        log.warning("Item '%s' has no image" % item.id)
                annotation += '%s %s\n' % (image_name, label)
            with open(annotation_file, 'w', encoding='utf-8') as f:
                f.write(annotation)

        labels_file = osp.join(subset_dir, ImagenetPath.LABELS_FILE)
        with open(labels_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(l.name
                for l in extractor.categories()[AnnotationType.label])
            )
