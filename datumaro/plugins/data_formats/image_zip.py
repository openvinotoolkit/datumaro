# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from enum import Enum
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA, ZIP_STORED, ZipFile

from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.exporter import Exporter
from datumaro.components.importer import Importer
from datumaro.components.media import ByteImage
from datumaro.util import parse_str_enum_value
from datumaro.util.image import IMAGE_EXTENSIONS, encode_image


class Compression(Enum):
    ZIP_STORED = ZIP_STORED
    ZIP_DEFLATED = ZIP_DEFLATED
    ZIP_BZIP2 = ZIP_BZIP2
    ZIP_LZMA = ZIP_LZMA


class ImageZipPath:
    DEFAULT_ARCHIVE_NAME = "default.zip"
    DEFAULT_COMPRESSION = Compression.ZIP_STORED


class ImageZipBase(SubsetBase):
    def __init__(self, url, subset=None, save_hash=False):
        super().__init__(subset=subset, media_type=ByteImage)

        assert url.endswith(".zip"), url

        with ZipFile(url, "r") as zf:
            for path in zf.filelist:
                item_id, extension = osp.splitext(path.filename)
                if extension.lower() not in IMAGE_EXTENSIONS:
                    continue
                image = ByteImage(data=zf.read(path.filename))
                self._items.append(
                    DatasetItem(id=item_id, media=image, subset=self._subset, save_hash=save_hash)
                )


class ImageZipImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, ".zip", "image_zip")


class ImageZipExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".jpg"

    @staticmethod
    def _get_compression_method(s):
        try:
            return Compression[s.upper()]
        except KeyError:
            import argparse

            raise argparse.ArgumentTypeError()

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)

        parser.add_argument(
            "--name",
            type=str,
            default=ImageZipPath.DEFAULT_ARCHIVE_NAME,
            help="Name of output zipfile (default: %(default)s)",
        )

        parser.add_argument(
            "--compression",
            type=cls._get_compression_method,
            default=ImageZipPath.DEFAULT_COMPRESSION.name,
            help="Archive compression method.\nAvailable methods: {} "
            "(default: %(default)s)".format(", ".join(e.name for e in Compression)),
        )

        return parser

    def __init__(self, extractor, save_dir, name=None, compression=None, **kwargs):
        super().__init__(extractor, save_dir, **kwargs)

        if name is None:
            name = ImageZipPath.DEFAULT_ARCHIVE_NAME

        compression = parse_str_enum_value(
            compression, Compression, default=ImageZipPath.DEFAULT_COMPRESSION
        )

        self._archive_name = name
        self._compression = compression.value

    def apply(self):
        os.makedirs(self._save_dir, exist_ok=True)

        archive_path = osp.join(self._save_dir, self._archive_name)

        if osp.exists(archive_path):
            raise FileExistsError(
                "Zip file: %s, already exist, "
                "specify archive name with --name extra argument" % archive_path
            )

        with ZipFile(archive_path, "w", self._compression) as zf:
            for item in self._extractor:
                if item.media:
                    self._archive_image(zf, item)
                else:
                    log.debug("Item '%s' has no image info", item.id)

    def _archive_image(self, zipfile, item):
        image_name = self._make_image_filename(item)
        if osp.isfile(item.media.path):
            zipfile.write(item.media.path, arcname=image_name)
        elif isinstance(item.media, ByteImage):
            zipfile.writestr(image_name, item.media.get_bytes())
        elif item.media.has_data:
            zipfile.writestr(image_name, encode_image(item.media.data, osp.splitext(image_name)[1]))
