# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import csv
import os.path as osp

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.util.image import find_images


class VottCsvPath:
    ANNO_FILE_SUFFIX = '-export.csv'

class VottCsvExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        super().__init__(subset=osp.splitext(osp.basename(path))[0].split('-')[0])

        self._categories = { AnnotationType.label: LabelCategories() }
        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        image_dir = osp.dirname(path)
        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace('\\', '/'): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        label_categories = self._categories[AnnotationType.label]

        with open(path, encoding='utf-8') as content:
            anno_table = list(csv.DictReader(content))
        for row in anno_table:
            item_id = osp.splitext(row['image'])[0]

            if item_id not in items:
                items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                    image=images.get(item_id))

            annotations = items[item_id].annotations
            if (len(row) == 6):
                label_name = row['label']
                label_idx = label_categories.find(label_name)[0]
                if label_idx is None:
                    label_idx = label_categories.add(label_name)

                x = float(row['xmin'])
                y = float(row['ymin'])
                w = float(row['xmax']) - x
                h = float(row['ymax']) - y

                annotations.append(Bbox(x, y, w, h, label=label_idx))

        return items

class VottCsvImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.csv', 'vott_csv')

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(f'*%s' % VottCsvPath.ANNO_FILE_SUFFIX)
