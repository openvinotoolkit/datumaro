
# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp

from datumaro.components.extractor import DatasetItem, SourceExtractor, Importer
from datumaro.components.converter import Converter
from datumaro.util.os_util import walk


class ImageDirImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        if not osp.isdir(path):
            return []
        return [{ 'url': path, 'format': 'image_dir' }]

class ImageDirExtractor(SourceExtractor):
    IMAGE_EXT_FORMATS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp',
        '.pgm', '.tif', '.tiff'}

    def __init__(self, url, max_depth=10):
        super().__init__()

        assert osp.isdir(url), url

        for dirpath, _, filenames in walk(url, max_depth=max_depth):
            for name in filenames:
                if not osp.splitext(name)[-1] in self.IMAGE_EXT_FORMATS:
                    continue
                path = osp.join(dirpath, name)
                item_id = osp.relpath(osp.splitext(path)[0], url)
                self._items.append(DatasetItem(id=item_id, image=path))

class ImageDirConverter(Converter):
    DEFAULT_IMAGE_EXT = '.jpg'

    def apply(self):
        os.makedirs(self._save_dir, exist_ok=True)

        for item in self._extractor:
            if item.has_image:
                self._save_image(item)
            else:
                log.debug("Item '%s' has no image info", item.id)