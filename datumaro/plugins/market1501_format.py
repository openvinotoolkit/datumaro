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
    IMAGE_NAMES = 'images_'

class Market1501Extractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isdir(path):
            raise NotADirectoryError("Can't open folder with annotation files '%s'" % path)
        subset = ''
        for dirname in glob(osp.join(path, '*')):
            if osp.basename(dirname).startswith(Market1501Path.BBOX_DIR):
                subset = osp.basename(dirname).replace(Market1501Path.BBOX_DIR, '')
            if osp.basename(dirname).startswith(Market1501Path.IMAGE_NAMES):
                subset = osp.basename(dirname).replace(Market1501Path.IMAGE_NAMES, '')
                subset = osp.splitext(subset)[0]
                break
        super().__init__(subset=subset)
        self._path = path
        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        paths = glob(osp.join(path, Market1501Path.QUERY_DIR, '*'))
        paths += glob(osp.join(path, Market1501Path.BBOX_DIR + self._subset, '*'))

        anno_file = osp.join(path, Market1501Path.IMAGE_NAMES + self._subset + '.txt')
        if len(paths) == 0 and osp.isfile(anno_file):
            with open(anno_file, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    paths.append(line)

        for image_path in paths:
            if osp.splitext(image_path)[-1] != Market1501Path.IMAGE_EXT:
                continue

            item_id = osp.splitext(osp.basename(image_path))[0]
            pid, camid = -1, -1
            search = Market1501Path.PATTERN.search(image_path)
            if search:
                pid, camid = map(int, search.groups())
                if 19 < len(item_id):
                    item_id = item_id[19:]
            items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                image=image_path)
            attributes = items[item_id].attributes
            if pid == -1:
                continue

            camid -= 1
            attributes['person_id'] = pid
            attributes['camera_id'] = camid
            if osp.basename(osp.dirname(image_path)) == Market1501Path.QUERY_DIR:
                attributes['query'] = True
            else:
                attributes['query'] = False
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
            annotation = ''
            for item in subset:
                image_name = item.id
                if Market1501Path.PATTERN.search(image_name) == None:
                    if 'person_id' in item.attributes and \
                            'camera_id' in item.attributes:
                        image_pattern = '{}{}_c{}s1_000000_00{}'
                        pid = int(item.attributes.get('person_id'))
                        camid = int(item.attributes.get('camera_id')) + 1
                        image_name = image_pattern.format('0' * (4 - len(str(pid))),
                            pid, camid, item.id)
                dirname = Market1501Path.BBOX_DIR + subset_name
                if 'query' in item.attributes and \
                        str(item.attributes.get('query')) == 'True':
                    dirname = Market1501Path.QUERY_DIR
                image_path = osp.join(self._save_dir, dirname,
                    image_name + Market1501Path.IMAGE_EXT)
                if item.has_image and self._save_images:
                    self._save_image(item, image_path)
                else:
                    annotation += '%s\n' % image_path
            if 0 < len(annotation):
                annotation_file = osp.join(self._save_dir,
                    Market1501Path.IMAGE_NAMES + subset_name + '.txt')
                with open(annotation_file, 'w') as f:
                    f.write(annotation)
