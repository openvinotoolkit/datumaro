# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json
import os.path as osp

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.media import Image
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class VottJsonPath:
    ANNO_FILE_SUFFIX = '-export.json'

class VottJsonExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        super().__init__(subset=osp.splitext(osp.basename(path))[0].split('-')[0])

        self._categories = { AnnotationType.label: LabelCategories() }
        if has_meta_file(path):
            self._categories = { AnnotationType.label: LabelCategories().
                from_iterable(list(parse_meta_file(path).keys())) }

        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        with open(path) as f:
            anno_dict = json.load(f)

        label_categories = self._categories[AnnotationType.label]
        tags = anno_dict.get('tags', [])
        for label in tags:
            label_name = label.get('name')
            if label_name:
                label_categories.add(label_name)

        for id, asset in anno_dict.get('assets', {}).items():
            item_id = osp.splitext(asset.get('asset').get('name'))[0]
            annotations = []
            for region in asset.get('regions', []):
                tags = region.get('tags', [])
                if not tags:
                    bbox = region['boundingBox']
                    if bbox:
                        annotations.append(Bbox(float(bbox['left']), float(bbox['top']),
                            float(bbox['width']), float(bbox['height']),
                            attributes={'id': region.get('id')}))

                for tag in region.get('tags', []):
                    label_idx = label_categories.find(tag)[0]
                    if label_idx is None:
                        label_idx = label_categories.add(tag)

                    bbox = region['boundingBox']
                    if bbox:
                        annotations.append(Bbox(float(bbox['left']), float(bbox['top']),
                            float(bbox['width']), float(bbox['height']), label=label_idx,
                            attributes={'id': region.get('id')}))

            items[item_id] = DatasetItem(id=item_id, subset=self._subset, attributes={'id': id},
                image=Image(path=osp.normpath(asset.get('asset').get('path'))),
                annotations=annotations)

        return items

class VottJsonImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.json', 'vott_json')

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(f'*%s' % VottJsonPath.ANNO_FILE_SUFFIX)
