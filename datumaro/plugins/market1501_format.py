# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from distutils.util import strtobool
from itertools import chain
import os
import os.path as osp
import re

from datumaro.components.converter import Converter
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.util.image import find_images


class Market1501Path:
    QUERY_DIR = 'query'
    BBOX_DIR = 'bounding_box_'
    IMAGE_EXT = '.jpg'
    PATTERN = re.compile(r'^(-?\d+)_c(\d+)(?:s\d+_\d+_00(.*))?')
    LIST_PREFIX = 'images_'
    UNKNOWN_ID = -1

class Market1501Extractor(SourceExtractor):
    def __init__(self, path, subset=None):
        if not osp.isdir(path):
            raise NotADirectoryError(
                "Can't open folder with annotation files '%s'" % path)

        if not subset:
            subset = ''
            for p in os.listdir(path):
                pf = osp.join(path, p)

                if p.startswith(Market1501Path.BBOX_DIR) and osp.isdir(pf):
                    subset = p.replace(Market1501Path.BBOX_DIR, '')
                    break

                if p.startswith(Market1501Path.LIST_PREFIX) and osp.isfile(pf):
                    subset = p.replace(Market1501Path.LIST_PREFIX, '')
                    subset = osp.splitext(subset)[0]
                    break
        super().__init__(subset=subset)

        self._path = path
        self._items = list(self._load_items(path).values())

    def _load_items(self, rootdir):
        items = {}

        paths = []
        anno_file = osp.join(rootdir,
            Market1501Path.LIST_PREFIX + self._subset + '.txt')
        if osp.isfile(anno_file):
            with open(anno_file, encoding='utf-8') as f:
                for line in f:
                    paths.append(osp.join(rootdir, line.strip()))
        else:
            paths = list(chain(
                find_images(osp.join(rootdir,
                        Market1501Path.QUERY_DIR),
                    recursive=True),
                find_images(osp.join(rootdir,
                        Market1501Path.BBOX_DIR + self._subset),
                    recursive=True),
            ))

        for image_path in paths:
            item_id = osp.splitext(osp.normpath(image_path))[0]
            if osp.isabs(image_path):
                item_id = osp.relpath(item_id, rootdir)
            subdir, item_id = item_id.split(os.sep, maxsplit=1)

            pid = Market1501Path.UNKNOWN_ID
            camid = Market1501Path.UNKNOWN_ID
            search = Market1501Path.PATTERN.search(osp.basename(item_id))
            if search:
                pid, camid = map(int, search.groups()[0:2])
                camid -= 1 # make ids 0-based
                custom_name = search.groups()[2]
                if custom_name:
                    item_id = osp.join(osp.dirname(item_id), custom_name)

            item = items.get(item_id)
            if item is None:
                item = DatasetItem(id=item_id, subset=self._subset,
                    image=image_path)
                items[item_id] = item

            if pid != Market1501Path.UNKNOWN_ID or \
                    camid != Market1501Path.UNKNOWN_ID:
                attributes = item.attributes
                attributes['query'] = subdir == Market1501Path.QUERY_DIR
                attributes['person_id'] = pid
                attributes['camera_id'] = camid
        return items

class Market1501Importer(Importer):
    @classmethod
    def find_sources(cls, path):
        if not osp.isdir(path):
            return []
        return [{ 'url': path, 'format': Market1501Extractor.NAME }]

class Market1501Converter(Converter):
    DEFAULT_IMAGE_EXT = Market1501Path.IMAGE_EXT

    def apply(self):
        for subset_name, subset in self._extractor.subsets().items():
            annotation = ''

            for item in subset:
                image_name = item.id
                if Market1501Path.PATTERN.search(image_name) is None:
                    if 'person_id' in item.attributes and \
                            'camera_id' in item.attributes:
                        image_pattern = '{:04d}_c{}s1_000000_00{}'
                        pid = int(item.attributes['person_id'])
                        camid = int(item.attributes['camera_id']) + 1
                        dirname, basename = osp.split(item.id)
                        image_name = osp.join(dirname,
                            image_pattern.format(pid, camid, basename))

                dirname = Market1501Path.BBOX_DIR + subset_name
                if 'query' in item.attributes:
                    query = item.attributes.get('query')
                    if isinstance(query, str):
                        query = strtobool(query)
                    if query:
                        dirname = Market1501Path.QUERY_DIR

                image_path = self._make_image_filename(item,
                    name=image_name, subdir=dirname)
                if self._save_images and item.has_image:
                    self._save_image(item, osp.join(self._save_dir, image_path))

                annotation += '%s\n' % image_path

            annotation_file = osp.join(self._save_dir,
                Market1501Path.LIST_PREFIX + subset_name + '.txt')
            with open(annotation_file, 'w', encoding='utf-8') as f:
                f.write(annotation)
