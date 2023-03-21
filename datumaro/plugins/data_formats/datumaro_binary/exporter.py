# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=no-self-use

import argparse
import logging as log
import os.path as osp
import struct
from io import BufferedWriter
from typing import Any, Optional

from datumaro.components.crypter import NULL_CRYPTER, Crypter
from datumaro.components.dataset_base import DatasetItem, IDataset
from datumaro.components.exporter import ExportContext, Exporter
from datumaro.plugins.data_formats.datumaro.exporter import DatumaroExporter
from datumaro.plugins.data_formats.datumaro.exporter import _SubsetWriter as __SubsetWriter
from datumaro.plugins.data_formats.datumaro.format import DATUMARO_FORMAT_VERSION
from datumaro.plugins.data_formats.datumaro_binary.mapper import DictMapper
from datumaro.plugins.data_formats.datumaro_binary.mapper.dataset_item import DatasetItemMapper

from .format import DatumaroBinaryPath


class _SubsetWriter(__SubsetWriter):
    """"""

    def __init__(
        self,
        context: Exporter,
        ann_file: str,
        secret_key_file: str,
        encryption_key: Optional[bytes] = None,
        no_media_encryption: bool = False,
    ):
        super().__init__(context, ann_file)
        self.secret_key_file = secret_key_file

        self._fp: Optional[BufferedWriter] = None
        self._crypter = Crypter(encryption_key) if encryption_key is not None else NULL_CRYPTER
        self._data["items"] = bytearray()
        self._item_cnt = 0
        media_type = context._extractor.media_type()
        self._media_type = {"media_type": media_type._type}

        self._media_encryption = not no_media_encryption

    def _sign(self):
        self._fp.write(DatumaroBinaryPath.SIGNATURE.encode())

    def _dump_encryption_field(self) -> int:
        if self._crypter.key is None:
            msg = b""
        else:
            msg = self._crypter.key
            msg = self._crypter.encrypt(msg)

        return self._fp.write(struct.pack(f"I{len(msg)}s", len(msg), msg))

    def _dump_header(self, header: Any, use_crypter: bool = True):
        msg = DictMapper.forward(header)

        if use_crypter and self._crypter.key is not None:
            msg = self._crypter.encrypt(msg)

        length = struct.pack("I", len(msg))
        return self._fp.write(length + msg)

    def _dump_version(self):
        self._dump_header(
            {
                "dm_format_version": DATUMARO_FORMAT_VERSION,
                "media_encryption": self._media_encryption,
            },
            use_crypter=False,
        )

    def _dump_info(self):
        self._dump_header(self.infos)

    def _dump_categories(self):
        self._dump_header(self.categories)

    def _dump_media_type(self):
        self._dump_header(self._media_type)

    def add_item(self, item: DatasetItem):
        with self.context_save_media(item, encryption=self._media_encryption):
            self.items.extend(DatasetItemMapper.forward(item))
        self._item_cnt += 1

    def _dump_items(self):
        items_bytes = self._crypter.encrypt(bytes(self.items))
        n_items_bytes = len(items_bytes)
        self._fp.write(struct.pack(f"I{n_items_bytes}s", self._item_cnt, items_bytes))

    def write(self):
        try:
            if not self._crypter.is_null_crypter:
                log.info(
                    "Please see the generated encryption secret key file in the following path.\n"
                    f"{self.secret_key_file}\n"
                    "It must be kept it separate from the dataset to protect your dataset safely. "
                    "You also need it to import the encrpted dataset in later, so that be careful not to lose."
                )

                with open(self.secret_key_file, "w") as fp:
                    fp.write(self._crypter.key.decode())

            with open(self.ann_file, "wb") as fp:
                self._fp = fp
                self._sign()
                self._dump_version()
                self._dump_encryption_field()
                self._dump_info()
                self._dump_categories()
                self._dump_media_type()
                self._dump_items()
        finally:
            self._fp = None


class EncryptionAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        encryption = True if option_string in self.option_strings else False
        if encryption:
            key = Crypter.gen_key()
        else:
            key = None

        setattr(namespace, "encryption_key", key)
        delattr(namespace, self.dest)


class DatumaroBinaryExporter(DatumaroExporter):
    DEFAULT_IMAGE_EXT = DatumaroBinaryPath.IMAGE_EXT
    PATH_CLS = DatumaroBinaryPath

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)

        parser.add_argument(
            "--encryption",
            action=EncryptionAction,
            default=False,
            help="Encrypt your dataset with the auto-generated secret key.",
        )

        parser.add_argument(
            "--no-media-encryption",
            action="store_true",
            help="Only encrypt the annotation file, not media files. "
            'This option is effective only if "--encryption" is enabled.',
        )

        return parser

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
        no_media_encryption: bool = False,
    ):
        self._encryption_key = encryption_key
        self._no_media_encryption = no_media_encryption
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
            secret_key_file=osp.join(self._save_dir, self.PATH_CLS.SECRET_KEY_FILE),
            encryption_key=self._encryption_key,
            no_media_encryption=self._no_media_encryption,
        )
