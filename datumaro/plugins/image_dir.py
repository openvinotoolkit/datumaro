
# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp

from datumaro.components.extractor import DatasetItem, SourceExtractor, Importer
from datumaro.components.converter import Converter
from datumaro.util.image import Image


class ImageDirImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        if not osp.isdir(path):
            return []
        return [{ 'url': path, 'format': 'image_dir' }]

class ImageDirExtractor(SourceExtractor):
    def __init__(self, url):
        super().__init__()

        assert osp.isdir(url), url

        for dirpath, _, filenames in os.walk(url):
            for name in filenames:
                path = osp.join(dirpath, name)
                image = Image(path=path)
                try:
                    # force loading
                    image.data # pylint: disable=pointless-statement
                except Exception:
                    continue

                item_id = osp.relpath(osp.splitext(path)[0], url)
                self._items.append(DatasetItem(id=item_id, image=image))

class ImageDirConverter(Converter):
    DEFAULT_IMAGE_EXT = '.jpg'

    def apply(self):
        os.makedirs(self._save_dir, exist_ok=True)

        for item in self._extractor:
            if item.has_image:
                self._save_image(item,
                    osp.join(self._save_dir, self._make_image_filename(item)))
            else:
                log.debug("Item '%s' has no image info", item.id)