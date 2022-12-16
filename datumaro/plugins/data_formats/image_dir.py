# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp

from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.exporter import Exporter
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.util.image import find_images


class ImageDirImporter(Importer):
    """
    Reads images from a directory as a dataset.
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--subset",
            help="The name of the subset for the produced dataset items " "(default: none)",
        )
        return parser

    @classmethod
    def find_sources(cls, path):
        if not osp.isdir(path):
            return []
        return [{"url": path, "format": ImageDirBase.NAME}]


class ImageDirBase(SubsetBase):
    def __init__(self, url, subset=None):
        super().__init__(subset=subset)

        assert osp.isdir(url), url

        for path in find_images(url, recursive=True):
            item_id = osp.relpath(osp.splitext(path)[0], url)
            self._items.append(
                DatasetItem(
                    id=item_id,
                    subset=self._subset,
                    media=Image(path=path),
                )
            )


class ImageDirExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".jpg"

    def apply(self):
        os.makedirs(self._save_dir, exist_ok=True)

        for item in self._extractor:
            if item.media:
                self._save_image(item)
            else:
                log.debug("Item '%s' has no image info", item.id)
