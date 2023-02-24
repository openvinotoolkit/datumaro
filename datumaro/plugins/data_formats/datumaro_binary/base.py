# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import struct
from io import BufferedWriter
from typing import Optional

from datumaro.components.errors import DatasetImportError
from datumaro.plugins.data_formats.datumaro_binary.format import DatumaroBinaryPath
from datumaro.util import parse_json

from ..datumaro.base import DatumaroBase
from .crypter import Crypter


class DatumaroBinaryBase(DatumaroBase):
    """"""

    def __init__(self, path: str, encryption_key: Optional[bytes] = None):
        self._fp: Optional[BufferedWriter] = None
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
        finally:
            self._fp = None

        return

    def _check_signature(self):
        signature = self._fp.read(DatumaroBinaryPath.SIGNATURE_LEN).decode()
        DatumaroBinaryPath.check_signature(signature)

    def _check_encryption_field(self):
        len_byte = self._fp.read(4)
        msg = self._fp.read(struct.unpack("I", len_byte)[0])

        # TODO: This will be developed later with encryption feature.
        extracted_key = self._crypter.decrypt(msg)
        if extracted_key == b"":
            extracted_key = None

        if extracted_key != self._crypter.key:
            raise DatasetImportError("Encryption key handshake fails. You may give a wrong key.")

    def _read_info(self):
        len_byte = self._fp.read(4)
        msg = self._fp.read(struct.unpack("I", len_byte)[0])
        msg = self._crypter.decrypt(msg)

        self._infos = parse_json(msg)

    def _read_categories(self):
        len_byte = self._fp.read(4)
        msg = self._fp.read(struct.unpack("I", len_byte)[0])
        msg = self._crypter.decrypt(msg)

        categories = parse_json(msg)
        self._categories = self._load_categories({"categories": categories})
