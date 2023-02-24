# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=no-self-use

import os.path as osp
import struct
from io import BufferedWriter
from typing import Any, Optional

from datumaro.components.dataset_base import IDataset
from datumaro.components.exporter import ExportContext
from datumaro.plugins.data_formats.datumaro.exporter import DatumaroExporter
from datumaro.plugins.data_formats.datumaro.exporter import _SubsetWriter as __SubsetWriter
from datumaro.plugins.data_formats.datumaro_binary.crypter import Crypter
from datumaro.util import dump_json

from .format import DatumaroBinaryPath


class _SubsetWriter(__SubsetWriter):
    """"""

    def __init__(self, context: IDataset, ann_file: str, encryption_key: Optional[bytes] = None):
        super().__init__(context, ann_file)
        self._fp: Optional[BufferedWriter] = None
        self._crypter = Crypter(encryption_key)

    def _sign(self):
        self._fp.write(DatumaroBinaryPath.SIGNATURE.encode())

    def _dump_encryption_field(self) -> int:
        if self._crypter.key is None:
            msg = b""
        else:
            msg = self._crypter.encrypt(self._crypter.key)

        length = struct.pack("I", len(msg))
        return self._fp.write(length + msg)

    def _dump_header(self, header: Any):
        msg = dump_json(header)

        if self._crypter.key is not None:
            msg = self._crypter.encrypt(msg)

        length = struct.pack("I", len(msg))
        return self._fp.write(length + msg)

    def _dump_info(self):
        self._dump_header(self.infos)

    def _dump_categories(self):
        self._dump_header(self.categories)

    def write(self):
        try:
            with open(self.ann_file, "wb") as fp:
                self._fp = fp
                self._sign()
                self._dump_encryption_field()
                self._dump_info()
                self._dump_categories()
        finally:
            self._fp = None


class DatumaroBinaryExporter(DatumaroExporter):
    DEFAULT_IMAGE_EXT = DatumaroBinaryPath.IMAGE_EXT
    PATH_CLS = DatumaroBinaryPath

    def __init__(
        self,
        extractor: IDataset,
        save_dir: str,
        *,
        save_images=None,
        save_media: Optional[bool] = None,
        image_ext: Optional[str] = None,
        default_image_ext: Optional[str] = None,
        save_dataset_meta: bool = False,
        ctx: Optional[ExportContext] = None,
        encryption_key: Optional[bytes] = None,
    ):
        self._encryption_key = encryption_key
        super().__init__(
            extractor,
            save_dir,
            save_images=save_images,
            save_media=save_media,
            image_ext=image_ext,
            default_image_ext=default_image_ext,
            save_dataset_meta=save_dataset_meta,
            ctx=ctx,
        )

    def create_writer(self, subset: str):
        return _SubsetWriter(
            context=self,
            ann_file=osp.join(self._annotations_dir, subset + self.PATH_CLS.ANNOTATION_EXT),
            encryption_key=self._encryption_key,
        )
