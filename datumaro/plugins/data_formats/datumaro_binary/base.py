# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import struct
from io import BufferedReader
from typing import Optional

from datumaro.components.errors import DatasetImportError
from datumaro.components.media import Image, MediaElement, MediaType, PointCloud
from datumaro.plugins.data_formats.datumaro_binary.format import DatumaroBinaryPath
from datumaro.plugins.data_formats.datumaro_binary.mapper import DictMapper
from datumaro.plugins.data_formats.datumaro_binary.mapper.dataset_item import DatasetItemMapper

from ..datumaro.base import DatumaroBase
from .crypter import Crypter


class DatumaroBinaryBase(DatumaroBase):
    """"""

    def __init__(self, path: str, encryption_key: Optional[bytes] = None):
        self._fp: Optional[BufferedReader] = None
        self._crypter = Crypter(encryption_key)
        super().__init__(path)

    def _load_impl(self, path: str) -> None:
        """Actual implementation of loading Datumaro binary format."""
        try:
            with open(path, "rb") as fp:
                self._fp = fp
                self._check_signature()
                self._check_encryption_field()
                self._read_info()
                self._read_categories()
                self._read_media_type()
                self._read_items()
        finally:
            self._fp = None

    def _check_signature(self):
        signature = self._fp.read(DatumaroBinaryPath.SIGNATURE_LEN).decode()
        DatumaroBinaryPath.check_signature(signature)

    def _check_encryption_field(self):
        len_byte = self._fp.read(4)
        _bytes = self._fp.read(struct.unpack("I", len_byte)[0])

        extracted_key = self._crypter.decrypt(_bytes)

        if not self._crypter.handshake(extracted_key):
            raise DatasetImportError("Encryption key handshake fails. You give a wrong key.")

    def _read_header(self):
        len_byte = self._fp.read(4)
        _bytes = self._fp.read(struct.unpack("I", len_byte)[0])
        _bytes = self._crypter.decrypt(_bytes)
        header, _ = DictMapper.backward(_bytes)
        return header

    def _read_info(self):
        self._infos = self._read_header()

    def _read_categories(self):
        categories = self._read_header()
        self._categories = self._load_categories({"categories": categories})

    def _read_media_type(self):
        media_type = self._read_header()["media_type"]
        if media_type == MediaType.IMAGE:
            self._media_type = Image
        elif media_type == MediaType.POINT_CLOUD:
            self._media_type = PointCloud
        elif media_type == MediaType.UNKNOWN:
            self._media_type = MediaElement
        else:
            raise NotImplementedError(f"media_type={media_type} is currently not supported.")

    def _read_items(self):
        (n_items,) = struct.unpack("I", self._fp.read(4))
        offset = 0
        _bytes = self._fp.read()

        self._items = []

        for _ in range(n_items):
            item, offset = DatasetItemMapper.backward(_bytes, offset)
            self._items.append(item)
