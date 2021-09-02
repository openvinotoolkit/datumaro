# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from tempfile import mkdtemp
from typing import Union
import logging as log
import os
import os.path as osp
import shutil

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset import DatasetPatch
from datumaro.components.extractor import DatasetItem
from datumaro.util.scope import scoped, on_error_do
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
    @scoped
    def patch(cls, dataset, patch, save_dir, **options):
        # This solution is not any better in performance than just
        # writing a dataset, but in case of patching (i.e. writing
        # to the previous location), it allows to avoid many problems
        # with removing and replacing existing files. Surely, this
        # approach also has problems with removal of the given directory.
        # Problems can occur if we can't remove the directory,
        # or want to reuse the given directory. It can happen if it
        # is mounted or (sym-)linked.
        # Probably, a better solution could be to wipe directory
        # contents and write new data there. Note that directly doing this
        # also doesn't work, because images may be needed for writing.

        if not osp.isdir(save_dir):
            return cls.convert(dataset, save_dir, **options)

        tmpdir = mkdtemp(dir=osp.dirname(save_dir),
            prefix=osp.basename(save_dir), suffix='.tmp')
        on_error_do(shutil.rmtree, tmpdir, ignore_errors=True)
        shutil.copymode(save_dir, tmpdir)

        retval = cls.convert(dataset, tmpdir, **options)

        shutil.rmtree(save_dir)
        os.replace(tmpdir, save_dir)

        return retval

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

        # TODO: refactor this variable.
        # Can be used by a subclass to store the current patch info
        if isinstance(extractor, DatasetPatch.DatasetPatchWrapper):
            self._patch = extractor.patch
        else:
            self._patch = None

    def _find_image_ext(self, item: Union[DatasetItem, Image]):
        src_ext = None

        if isinstance(item, DatasetItem) and item.has_image:
            src_ext = item.image.ext
        elif isinstance(item, Image):
            src_ext = item.ext

        return self._image_ext or src_ext or self._default_image_ext

    def _make_item_filename(self, item, *, name=None, subdir=None):
        name = name or item.id
        subdir = subdir or ''
        return osp.join(subdir, name)

    def _make_image_filename(self, item, *, name=None, subdir=None):
        return self._make_item_filename(item, name=name, subdir=subdir) + \
            self._find_image_ext(item)

    def _make_pcd_filename(self, item, *, name=None, subdir=None):
        return self._make_item_filename(item, name=name, subdir=subdir) + '.pcd'

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

    def _save_point_cloud(self, item=None, path=None, *,
            name=None, subdir=None, basedir=None):
        assert not ((subdir or name or basedir) and path), \
            "Can't use both subdir or name or basedir and path arguments"

        if not item.point_cloud:
            log.warning("Item '%s' has no pcd", item.id)
            return

        basedir = basedir or self._save_dir
        path = path or osp.join(basedir,
            self._make_pcd_filename(item, name=name, subdir=subdir))
        path = osp.abspath(path)

        os.makedirs(osp.dirname(path), exist_ok=True)
        if item.point_cloud and osp.isfile(item.point_cloud):
            if item.point_cloud != path:
                shutil.copyfile(item.point_cloud, path)
