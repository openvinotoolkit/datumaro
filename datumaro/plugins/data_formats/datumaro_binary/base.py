# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import struct
from io import BufferedReader
from multiprocessing.pool import AsyncResult, Pool
from typing import Any, Dict, List, Optional

from datumaro.components.crypter import NULL_CRYPTER, Crypter
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import DatasetImportError
from datumaro.components.media import Image, MediaElement, MediaType, PointCloud
from datumaro.plugins.data_formats.datumaro_binary.format import DatumaroBinaryPath
from datumaro.plugins.data_formats.datumaro_binary.mapper import DictMapper
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import IntListMapper
from datumaro.plugins.data_formats.datumaro_binary.mapper.dataset_item import DatasetItemMapper

from ..datumaro.base import DatumaroBase


class DatumaroBinaryBase(DatumaroBase):
    """"""

    def __init__(self, path: str, encryption_key: Optional[bytes] = None, num_workers: int = 0):
        """
        Parameters
        ----------
        path
            Directory path to import DatumaroBinary format dataset
        encryption_key
            If the dataset is encrypted, it (secret key) is needed to import the dataset.
        num_workers
            The number of multi-processing workers for import. If num_workers = 0, do not use multiprocessing.
        """
        self._fp: Optional[BufferedReader] = None
        self._crypter = Crypter(encryption_key) if encryption_key is not None else NULL_CRYPTER
        self._media_encryption = False
        self._num_workers = num_workers
        super().__init__(path)

    def _get_dm_format_version(self, path: str) -> str:
        with open(path, "rb") as fp:
            self._fp = fp
            self._check_signature()
            dm_format_version = self._read_version()
        return dm_format_version

    def _load_impl(self, path: str) -> None:
        """Actual implementation of loading Datumaro binary format."""
        try:
            with open(path, "rb") as fp:
                self._fp = fp
                self._check_signature()
                self._read_version()
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

        if not self._crypter.handshake(_bytes):
            raise DatasetImportError("Encryption key handshake fails. You give a wrong key.")

    def _read_header(self, use_crypter: bool = True):
        len_byte = self._fp.read(4)
        _bytes = self._fp.read(struct.unpack("I", len_byte)[0])
        if use_crypter:
            _bytes = self._crypter.decrypt(_bytes)
        header, _ = DictMapper.backward(_bytes)
        return header

    def _read_version(self) -> Dict[str, Any]:
        version_header = self._read_header(use_crypter=False)
        self._media_encryption = version_header["media_encryption"]
        return version_header["dm_format_version"]

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
        elif media_type == MediaType.MEDIA_ELEMENT:
            self._media_type = MediaElement
        else:
            raise NotImplementedError(f"media_type={media_type} is currently not supported.")

    def _read_items(self) -> None:
        (n_blob_sizes_bytes,) = struct.unpack("<I", self._fp.read(4))
        blob_sizes_bytes = self._crypter.decrypt(self._fp.read(n_blob_sizes_bytes))
        blob_sizes, _ = IntListMapper.backward(blob_sizes_bytes, 0)

        media_path_prefix = {
            MediaType.IMAGE: osp.join(self._images_dir, self._subset),
            MediaType.POINT_CLOUD: osp.join(self._pcd_dir, self._subset),
        }

        if self._num_workers > 0:
            self._items = self._read_items_mp(blob_sizes, media_path_prefix)
        else:
            self._items = self._read_items_sp(blob_sizes, media_path_prefix)

        for item in self._items:
            if item.media is not None and self._media_encryption:
                item.media.set_crypter(self._crypter)

    def _read_items_mp(
        self, blob_sizes: List[int], media_path_prefix: Dict[MediaType, str]
    ) -> List[DatasetItem]:
        async_results: List[AsyncResult] = []

        with Pool(processes=self._num_workers) as pool:
            for blob_size in blob_sizes:
                blob_bytes = self._fp.read(blob_size)
                async_results += [
                    pool.apply_async(
                        self._read_blob,
                        (
                            blob_bytes,
                            self._crypter,
                            media_path_prefix,
                        ),
                    )
                ]

            return [
                item
                for async_result in async_results
                for item in async_result.get(timeout=DatumaroBinaryPath.MP_TIMEOUT)
            ]

    def _read_items_sp(
        self, blob_sizes: List[int], media_path_prefix: Dict[MediaType, str]
    ) -> List[DatasetItem]:
        items_list = [
            self._read_blob(self._fp.read(blob_size), self._crypter, media_path_prefix)
            for blob_size in blob_sizes
        ]

        return [item for items in items_list for item in items]

    @staticmethod
    def _read_blob(
        blob_bytes: bytes, crypter: Crypter, media_path_prefix: Dict[MediaType, str]
    ) -> List[DatasetItem]:
        items = []
        offset = 0

        # Decrypt bytes first
        blob_bytes = crypter.decrypt(blob_bytes)

        # Extract items
        while offset < len(blob_bytes):
            item, offset = DatasetItemMapper.backward(blob_bytes, offset, media_path_prefix)
            items.append(item)

        assert offset == len(blob_bytes)

        return items
