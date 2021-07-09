
# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp

from datumaro.components.converter import Converter
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.util.image import find_images


class ImageDirImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        if not osp.isdir(path):
            return []
        return [{ 'url': path, 'format': 'image_dir' }]

class ImageDirExtractor(SourceExtractor):
    def __init__(self, url, subset=None, max_depth=None, exts=None):
        super().__init__(subset=subset)

        assert osp.isdir(url), url

        for path in find_images(url, exts=exts,
                recursive=True, max_depth=max_depth):
            item_id = osp.relpath(osp.splitext(path)[0], url)
            self._items.append(DatasetItem(id=item_id, subset=self._subset,
                image=path))

class ImageDirConverter(Converter):
    DEFAULT_IMAGE_EXT = '.jpg'

    def apply(self):
        os.makedirs(self._save_dir, exist_ok=True)

        for item in self._extractor:
            if item.has_image:
                self._save_image(item)
            else:
                log.debug("Item '%s' has no image info", item.id)