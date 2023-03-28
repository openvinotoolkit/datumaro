# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import datetime
import os
from copy import deepcopy
from typing import Callable, Optional, Union

import pyarrow as pa
import pytz
from tqdm import tqdm

from datumaro.components.crypter import NULL_CRYPTER
from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetItem, IDataset
from datumaro.components.exporter import ExportContext, ExportContextComponent, Exporter
from datumaro.plugins.data_formats.datumaro.exporter import _SubsetWriter as __SubsetWriter
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import DictMapper

from .format import DatumaroArrow
from .mapper.dataset_item import DatasetItemMapper
from .mapper.media import ImageFileMapper


class _SubsetWriter(__SubsetWriter):
    def __init__(
        self,
        context: Exporter,
        export_context: ExportContextComponent,
        subset: str,
        writer_batch_size: int = 1000,
    ):
        super().__init__(context, "", export_context)
        self._schema = deepcopy(DatumaroArrow.SCHEMA)
        self._subset = subset
        self._writer = None
        self._writer_batch_size = writer_batch_size

        self._data = {
            "items": [],
            "infos": {},
            "categories": {},
            "built_time": str(datetime.datetime.now(pytz.utc)),
            "source_path": self.export_context.source_path,
            "version": str(DatumaroArrow.VERSION),
            "signature": DatumaroArrow.SIGNATURE,
        }

    def add_infos(self, infos):
        if self._writer is not None:
            raise ValueError("Writer has been initialized.")
        super().add_infos(infos)
        self._data["infos"] = DictMapper.forward(self.infos)

    def add_categories(self, categories):
        if self._writer is not None:
            raise ValueError("Writer has been initialized.")
        super().add_categories(categories)
        self._data["categories"] = DictMapper.forward(self.categories)

    def init_writer(self):
        self._schema = self._schema.with_metadata(
            {k: v for k, v in self._data.items() if k != "items"}
        )

        f_name = os.path.join(self._context._save_dir, f"{self._subset}.arrow")
        self._writer = pa.RecordBatchStreamWriter(f_name, self._schema)

    def add_item(self, item: DatasetItem):
        item = DatasetItemMapper.forward(item, media={"encoder": self._context._image_ext})
        if item["media"]["bytes"] is not None:
            item["media"]["path"] = item["media"]["path"].replace(
                self.export_context.source_path, ""
            )
        self.items.append(item)

    def write(self, writer_batch_size: Optional[int] = None):
        if writer_batch_size is None:
            writer_batch_size = self._writer_batch_size
            if len(self.items) < writer_batch_size:
                return
        assert self._writer is not None
        for i in range(0, len(self.items), writer_batch_size):
            batch = self.items[i : i + writer_batch_size]
            arrays = []
            for name in self._schema.names:
                arrays.append([data[name] for data in batch])
            pa_table = pa.Table.from_arrays(arrays, schema=self._schema)
            self._writer.write_table(pa_table, writer_batch_size)
        self._data["items"] = []


class ArrowExporter(Exporter):
    AVAILABLE_IMAGE_EXTS = ImageFileMapper.AVAILABLE_SCHEMES
    DEFAULT_IMAGE_EXT = ImageFileMapper.AVAILABLE_SCHEMES[0]

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)

        parser.add_argument(
            "--image-ext",
            default=None,
            help=f"Image encoding scheme (default: {cls.DEFAULT_IMAGE_EXT})",
            choices=cls.AVAILABLE_IMAGE_EXTS,
        )

        return parser

    def create_writer(self, subset: str) -> _SubsetWriter:
        export_context = ExportContextComponent(
            save_dir=self._save_dir,
            save_media=self._save_media,
            images_dir="",
            pcd_dir="",
            related_images_dir="",
            crypter=NULL_CRYPTER,
            image_ext=self._image_ext,
            default_image_ext=self._default_image_ext,
            source_path=self._extractor._source_path
            if getattr(self._extractor, "_source_path")
            else None,
        )

        return _SubsetWriter(
            context=self,
            subset=subset,
            export_context=export_context,
        )

    def apply(self):
        os.makedirs(self._save_dir, exist_ok=True)

        writers = {subset: self.create_writer(subset) for subset in self._extractor.subsets()}

        for writer in writers.values():
            writer.add_infos(self._extractor.infos())
            writer.add_categories(self._extractor.categories())
            writer.init_writer()

        for item in tqdm(self._extractor):
            subset = item.subset or DEFAULT_SUBSET_NAME
            writers[subset].add_item(item)
            writers[subset].write()

        for writer in writers.values():
            writer.write(writer._writer_batch_size)

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        raise NotImplementedError

    def __init__(
        self,
        extractor: IDataset,
        save_dir: str,
        *,
        save_images=None,  # Deprecated
        save_media: Optional[bool] = None,
        image_ext: Optional[Union[str, Callable[[str], bytes]]] = None,
        default_image_ext: Optional[str] = None,
        save_dataset_meta: bool = False,
        ctx: Optional[ExportContext] = None,
    ):
        super().__init__(
            extractor=extractor,
            save_dir=save_dir,
            save_images=save_images,
            save_media=save_media,
            image_ext=image_ext,
            default_image_ext=default_image_ext,
            save_dataset_meta=save_dataset_meta,
            ctx=ctx,
        )

        if self._save_media:
            self._image_ext = (
                self._image_ext if self._image_ext is not None else self._default_image_ext
            )
        else:
            self._image_ext = "NONE"

        assert (
            self._image_ext in self.AVAILABLE_IMAGE_EXTS
        ), f"{self._image_ext} is unkonwn ext. Available exts are {self.AVAILABLE_IMAGE_EXTS}"
