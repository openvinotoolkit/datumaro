# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Union
import logging as log
import os
import os.path as osp
import shutil

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.extractor import DatasetItem
from datumaro.util.image import Image


class Converter(CliPlugin):
    DEFAULT_IMAGE_EXT = None

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('--save-images', action='store_true',
            help="Save images (default: %(default)s)")
        parser.add_argument('--image-ext', default=None,
            help="Image extension (default: keep or use format default%s)" % \
                (' ' + cls.DEFAULT_IMAGE_EXT if cls.DEFAULT_IMAGE_EXT else ''))

        return parser

    @classmethod
    def convert(cls, extractor, save_dir, **options):
        converter = cls(extractor, save_dir, **options)
        return converter.apply()

    @classmethod
    def patch(cls, dataset, patch, save_dir, **options):
        return cls.convert(dataset, save_dir, **options)

    def apply(self):
        raise NotImplementedError("Should be implemented in a subclass")

    def __init__(self, extractor, save_dir, save_images=False,
            image_ext=None, default_image_ext=None):
        default_image_ext = default_image_ext or self.DEFAULT_IMAGE_EXT
        assert default_image_ext
        self._default_image_ext = default_image_ext

        self._save_images = save_images
        self._image_ext = image_ext

        self._extractor = extractor
        self._save_dir = save_dir

    def _find_image_ext(self, item: Union[DatasetItem, Image]):
        src_ext = None

        if isinstance(item, DatasetItem) and item.has_image:
            src_ext = item.image.ext
        elif isinstance(item, Image):
            src_ext = item.ext

        return self._image_ext or src_ext or self._default_image_ext

    def _make_image_filename(self, item, *, name=None, subdir=None):
        name = name or item.id
        subdir = subdir or ''
        return osp.join(subdir, name + self._find_image_ext(item))

    def _make_pcd_filename(self, item, *, name=None, subdir=None):
        name = name or item.id
        subdir = subdir or ''
        return osp.join(subdir, name + '.pcd')

    def _save_image(self, item, path=None, *,
            name=None, subdir=None, basedir=None):
        assert not ((subdir or name or basedir) and path), \
            "Can't use both subdir or name or basedir and path arguments"

        if not item.has_image or not item.image.has_data:
            log.warning("Item '%s' has no image", item.id)
            return

        basedir = basedir or self._save_dir
        path = path or osp.join(basedir,
            self._make_image_filename(item, name=name, subdir=subdir))
        path = osp.abspath(path)

        item.image.save(path)

    def _save_pcd(self, item=None, path=None, *,
            name=None, subdir=None, basedir=None):
        assert not ((subdir or name or basedir) and path), \
            "Can't use both subdir or name or basedir and path arguments"

        if not item.pcd:
            log.warning("Item '%s' has no pcd", item.id)
            return

        basedir = basedir or self._save_dir
        path = path or osp.join(basedir,
            self._make_pcd_filename(item, name=name, subdir=subdir))
        path = osp.abspath(path)

        os.makedirs(osp.dirname(path), exist_ok=True)
        if item.pcd and osp.isfile(item.pcd):
            if item.pcd != path:
                shutil.copyfile(item.pcd, path)
        elif isinstance(item.pcd, bytes):
            with open(path, 'wb') as f:
                f.write(item.pcd)
