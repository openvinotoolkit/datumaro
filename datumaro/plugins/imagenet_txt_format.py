
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from glob import glob

from datumaro.components.extractor import (DatasetItem, Label,
    LabelCategories, AnnotationType, SourceExtractor, Importer
)
from datumaro.components.converter import Converter


class ImagenetTxtPath:
    LABELS_FILE = 'synsets.txt'

class ImagenetTxtExtractor(SourceExtractor):
    def __init__(self, path):
        assert osp.isfile(path), path
        super().__init__(subset=osp.splitext(osp.basename(path))[0])

        labels = osp.join(osp.dirname(path), ImagenetTxtPath.LABELS_FILE)
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
        with open(path, encoding='utf-8') as f:
            for line in f:
                item = line.split()
                item_id = item[0]
                labels_id = item[1:]
                anno = []
                for label_id in labels_id:
                    anno += [Label(label=label_id)]
                items[item_id] = DatasetItem(
                    id=item_id, subset=self._subset,
                    annotations=anno
                )
        return items


class ImagenetTxtImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        subset_paths = [p for p in glob(osp.join(path, '*.txt'))
            if 'synsets' not in osp.basename(p)]
        sources = []
        for subset_path in subset_paths:
            sources += cls._find_sources_recursive(subset_path, '.txt', 'imagenet_txt')
        return sources


class ImagenetTxtConverter(Converter):
    DEFAULT_IMAGE_EXT = '.jpg'

    def apply(self):
        subset_dir = self._save_dir
        extractor = self._extractor
        for subset_name, subset in self._extractor.subsets().items():
            annotation_file = osp.join(subset_dir, '%s.txt' % subset_name)
            labels = {}
            for item in subset:
                labels[item.id] = [str(p.label) for p in item.annotations
                    if p.type == AnnotationType.label]

            with open(annotation_file, 'w', encoding='utf-8') as f:
                f.writelines(['%s %s\n' % (item_id, ' '.join(labels[item_id])) for item_id in labels])

        labels_file = osp.join(subset_dir, ImagenetTxtPath.LABELS_FILE)
        with open(labels_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(l.name
                for l in extractor.categories()[AnnotationType.label])
            )
