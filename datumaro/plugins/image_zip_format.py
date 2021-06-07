# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA, ZIP_STORED, ZipFile

from datumaro.components.converter import Converter
from datumaro.components.extractor import (DatasetItem, Importer,
                                           SourceExtractor)
from datumaro.util.image import (IMAGE_EXTENSIONS, ByteImage, decode_image,
                                 encode_image)

class ImageZipPath:
    DEFAULT_ARCHIVE_NAME = 'default.zip'
    DEFAULT_COMPRESSION = ZIP_STORED

    COMPRESSION = {
       'ZIP_STORED': ZIP_STORED,
       'ZIP_DEFLATED': ZIP_DEFLATED,
       'ZIP_BZIP2': ZIP_BZIP2,
       'ZIP_LZMA': ZIP_LZMA
    }

class ImageZipExtractor(SourceExtractor):
    def __init__(self, url, subset=None):
        super().__init__(subset=subset)

        assert url.endswith('.zip'), url

        with ZipFile(url, 'r') as zf:
            for path in zf.filelist:
                item_id, extension = osp.splitext(path.filename)
                if extension.lower() not in IMAGE_EXTENSIONS:
                    continue
                print(item_id)
                image = decode_image(zf.read(path.filename))
                self._items.append(DatasetItem(
                    id=item_id, image=image, subset=self._subset
                ))

class ImageZipImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.zip', 'image_zip')

class ImageZipConverter(Converter):
    DEFAULT_IMAGE_EXT = '.jpg'

    @staticmethod
    def _get_compression_method(s):
        try:
            return ImageZipPath.COMPRESSION[s.upper()]
        except KeyError:
            import argparse
            raise argparse.ArgumentTypeError()

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)

        parser.add_argument('--name', type=str,
            default=ImageZipPath.DEFAULT_ARCHIVE_NAME,
            help="Name of output zipfile "
                "(default: %(default)s)"
        )

        parser.add_argument('--compression', type=cls._get_compression_method,
            default=ImageZipPath.DEFAULT_COMPRESSION,
            help="Archive compression method.\nAvailable methods: %s." % \
                 ', '.join(list(ImageZipPath.COMPRESSION.keys()))
        )

        return parser

    def __init__(self, extractor, save_dir, name=None,
            compression=None, **kwargs):
        super().__init__(extractor, save_dir, **kwargs)

        if name is None:
            name = ImageZipPath.DEFAULT_ARCHIVE_NAME
        if compression is None:
            compression = ImageZipPath.DEFAULT_COMPRESSION

        self._archive_name = name
        self._compression = compression

    def apply(self):
        os.makedirs(self._save_dir, exist_ok=True)

        archive_path = osp.join(self._save_dir, self._archive_name)

        if osp.exists(archive_path):
            raise FileExistsError('Zip file: %s, already exist,'
                'specify archive name with --name extra argument' % archive_path)

        with ZipFile(archive_path, 'w', self._compression) as zf:
            for item in self._extractor:
                if item.has_image:
                    self._archive_image(zf, item)
                else:
                    log.debug("Item '%s' has no image info", item.id)

    def _archive_image(self, zipfile, item):
        image_name = self._make_image_filename(item)
        if osp.isfile(item.image.path):
            zipfile.write(item.image.path, arcname=image_name)
        elif isinstance(item.image, ByteImage):
            zipfile.writestr(image_name, item.image.get_bytes())
        elif item.image.has_data:
            zipfile.writestr(image_name,
                encode_image(item.image.data, osp.splitext(image_name)[1]))
