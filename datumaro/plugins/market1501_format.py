# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import re
from glob import glob

from datumaro.components.converter import Converter
from datumaro.components.extractor import (DatasetItem, Importer,
    SourceExtractor)


class Market1501Path:
    QUERY_DIR = 'query'
    BBOX_DIR = 'bounding_box_'
    IMAGE_EXT = '.jpg'
    PATTERN = re.compile(r'([-\d]+)_c(\d)')

class Market1501Extractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isdir(path):
            raise NotADirectoryError("Can't open folder with annotation files '%s'" % path)
        subset = ''
        for dirname in glob(osp.join(path, '*')):
            if osp.basename(dirname).startswith(Market1501Path.BBOX_DIR):
                subset = osp.basename(dirname).replace(Market1501Path.BBOX_DIR, '')
                break
        super().__init__(subset=subset)
        self._path = path
        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        paths = glob(osp.join(path, Market1501Path.QUERY_DIR, '*'))
        paths += glob(osp.join(path, Market1501Path.BBOX_DIR + self._subset, '*'))

        for image_path in paths:
            if not osp.isfile(image_path) or \
                    osp.splitext(image_path)[-1] != Market1501Path.IMAGE_EXT:
                continue

            item_id = osp.splitext(osp.basename(image_path))[0]
            attributes = {}
            pid, camid = map(int, Market1501Path.PATTERN.search(image_path).groups())
            if pid == -1:
                continue

            camid -= 1
            attributes['person_id'] = pid
            attributes['camera_id'] = camid
            if osp.basename(osp.dirname(image_path)) == Market1501Path.QUERY_DIR:
                attributes['query'] = True
            else:
                attributes['query'] = False
            items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                image=image_path, attributes=attributes)
        return items

class Market1501Importer(Importer):
    @classmethod
    def find_sources(cls, path):
        if not osp.isdir(path):
            return []
        return [{ 'url': path, 'format': 'market1501' }]

class Market1501Converter(Converter):
    DEFAULT_IMAGE_EXT = '.jpg'

    def apply(self):
        for subset_name, subset in self._extractor.subsets().items():
            for item in subset:
                if item.has_image and self._save_images:
                    if item.attributes and 'query' in item.attributes:
                        if query in item.attributes:
                            dirname = Market1501Path.QUERY_DIR
                        else:
                            dirname = Market1501Path.BBOX_DIR + subset_name
                        self._save_image(item, osp.join(self._save_dir,
                            dirname, item.id + Market1501Path.IMAGE_EXT))
