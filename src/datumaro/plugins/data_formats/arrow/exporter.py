# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
import datetime
import os
import platform
import re
import struct
import tempfile
from copy import deepcopy
from functools import partial
from multiprocessing.pool import ApplyResult, AsyncResult, Pool
from shutil import move, rmtree
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import memory_profiler
import numpy as np
import pyarrow as pa
import pytz

from datumaro.components.crypter import NULL_CRYPTER
from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetItem, IDataset
from datumaro.components.errors import DatumaroError
from datumaro.components.exporter import ExportContext, ExportContextComponent, Exporter
from datumaro.plugins.data_formats.datumaro.exporter import _SubsetWriter as __SubsetWriter
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import DictMapper
from datumaro.util.file_utils import to_bytes
from datumaro.util.multi_procs_util import consumer_generator

from .format import DatumaroArrow
from .mapper.dataset_item import DatasetItemMapper
from .mapper.media import ImageMapper


class ArrowExporter(Exporter):
    AVAILABLE_IMAGE_EXTS = ImageMapper.AVAILABLE_SCHEMES
    DEFAULT_IMAGE_EXT = ImageMapper.AVAILABLE_SCHEMES[0]

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)

        # '--image-ext' would be used in a different way for arrow foramt
        _actions = []
        for action in parser._actions:
            if action.dest != "image_ext":
                _actions.append(action)
        parser._actions = _actions
        parser._option_string_actions.pop("--image-ext")

        parser.add_argument(
            "--image-ext",
            default=None,
            help=f"Image encoding scheme. (default: {cls.DEFAULT_IMAGE_EXT})",
            choices=cls.AVAILABLE_IMAGE_EXTS,
        )

        parser.add_argument(
            "--max-chunk-size",
            type=int,
            default=1000,
            help="The maximum number of dataset item can be stored in each shard file. "
            "'--max-chunk-size' and '--num-shards' are mutually exclusive, "
            "Therefore, if '--max-chunk-size' is not None, the number of shard files will be determined by "
            "(# of total dataset item) / (max chunk size). "
            "(default: %(default)s)",
        )

        parser.add_argument(
            "--num-shards",
            type=int,
            default=None,
            help="The number of shards to export. "
            "'--max-chunk-size' and '--num-shards' are mutually exclusive. "
            "Therefore, if '--num-shards' is not None, the number of dataset item in each shard file "
            "will be determined by (# of total dataset item) / (num shards). "
            "(default: %(default)s)",
        )

        parser.add_argument(
            "--num-workers",
            type=int,
            default=0,
            help="The number of multi-processing workers for export. "
            "If num_workers = 0, do not use multiprocessing. (default: %(default)s)",
        )

        parser.add_argument(
            "--prefix",
            type=str,
            default="datum",
            help="Prefix to be appended in front of the shard file name. "
            "Therefore, the generated file name will be `<prefix>-<idx>.arrow`. "
            "(default: %(default)s)",
        )

        return parser

    def _apply_impl(self, *args, **kwargs):
        if self._num_workers == 0:
            return self._apply()

        with Pool(processes=self._num_workers) as pool:
            return self._apply(pool)

    def _apply(self, pool: Optional[Pool] = None):
        os.makedirs(self._save_dir, exist_ok=True)

        if pool is not None:

            def _producer_gen():
                for item in self._extractor:
                    future = pool.apply_async(
                        func=self._item_to_dict_record,
                        args=(item, self._image_ext, self._source_path),
                    )
                    yield future

            with consumer_generator(producer_generator=_producer_gen()) as consumer_gen:
                self._write_file(consumer_gen)

        else:

            def create_consumer_gen():
                for item in self._extractor:
                    yield self._item_to_dict_record(item, self._image_ext, self._source_path)

            self._write_file(create_consumer_gen())

    def _write_file(self, consumer_gen: Iterator[Dict[str, Any]]) -> None:
        for file_idx, size in enumerate(self._chunk_sizes):
            record_batch = pa.RecordBatch.from_pylist(
                mapping=[next(consumer_gen) for _ in range(size)],
                schema=self._schema,
            )

            suffix = str(file_idx).zfill(self._max_digits)
            fpath = os.path.join(self._save_dir, f"{self._prefix}-{suffix}.arrow")

            with pa.OSFile(fpath, "wb") as sink:
                with pa.ipc.new_file(sink, self._schema) as writer:
                    writer.write(record_batch)

        pass

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        # no patch supported
        with tempfile.TemporaryDirectory() as temp_dir:
            cls.convert(dataset, save_dir=temp_dir, **kwargs)
            if os.path.exists(save_dir):
                for file in os.listdir(save_dir):
                    file = os.path.join(save_dir, file)
                    if os.path.isdir(file):
                        rmtree(file)
                    else:
                        os.remove(file)
            for file in os.listdir(temp_dir):
                file_from = os.path.join(temp_dir, file)
                file_to = os.path.join(save_dir, file)
                move(file_from, file_to)

    def __init__(
        self,
        extractor: IDataset,
        save_dir: str,
        *,
        save_media: Optional[bool] = None,
        image_ext: Optional[Union[str, Callable[[str], bytes]]] = None,
        default_image_ext: Optional[str] = None,
        save_dataset_meta: bool = False,
        ctx: Optional[ExportContext] = None,
        num_workers: int = 0,
        max_chunk_size: Optional[int] = 1000,
        num_shards: Optional[int] = None,
        prefix: str = "datum",
        **kwargs,
    ):
        super().__init__(
            extractor=extractor,
            save_dir=save_dir,
            save_media=save_media,
            image_ext=image_ext,
            default_image_ext=default_image_ext,
            save_dataset_meta=save_dataset_meta,
            ctx=ctx,
        )

        if num_workers < 0:
            raise DatumaroError(
                f"num_workers should be non-negative but num_workers={num_workers}."
            )
        self._num_workers = num_workers

        if num_shards is not None and max_chunk_size is not None:
            raise DatumaroError(
                "Both 'num_shards' or 'max_chunk_size' cannot be provided at the same time."
            )
        elif num_shards is not None and num_shards < 0:
            raise DatumaroError(f"num_shards should be non-negative but num_shards={num_shards}.")
        elif max_chunk_size is not None and max_chunk_size < 0:
            raise DatumaroError(
                f"max_chunk_size should be non-negative but max_chunk_size={max_chunk_size}."
            )
        elif num_shards is None and max_chunk_size is None:
            raise DatumaroError(
                "Either one of 'num_shards' or 'max_chunk_size' should be provided."
            )

        self._num_shards = num_shards
        self._max_chunk_size = max_chunk_size

        if self._save_media:
            self._image_ext = (
                self._image_ext if self._image_ext is not None else self._default_image_ext
            )
        else:
            self._image_ext = "NONE"

        assert (
            self._image_ext in self.AVAILABLE_IMAGE_EXTS
        ), f"{self._image_ext} is unkonwn ext. Available exts are {self.AVAILABLE_IMAGE_EXTS}"

        self._prefix = prefix

        self._source_path = (
            os.path.abspath(self._extractor._source_path)
            if getattr(self._extractor, "_source_path")
            else None
        )

        total_len = len(self._extractor)

        if self._num_shards is not None:
            max_chunk_size = int(total_len / self._num_shards) + 1

        elif self._max_chunk_size is not None:
            max_chunk_size = self._max_chunk_size
        else:
            raise DatumaroError(
                "Either one of 'num_shards' or 'max_chunk_size' should be provided."
            )

        self._chunk_sizes = np.diff(
            np.array([size for size in range(0, total_len, max_chunk_size)] + [total_len])
        )
        assert (
            sum(self._chunk_sizes) == total_len
        ), "Sum of chunk sizes should be the number of total items."

        num_shard_files = len(self._chunk_sizes)
        self._max_digits = len(str(num_shard_files))

        self._schema = DatumaroArrow.create_schema_with_metadata(self._extractor)

    @staticmethod
    def _item_to_dict_record(
        item: DatasetItem,
        image_ext: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        dict_item = DatasetItemMapper.forward(item, media={"encoder": image_ext})

        if dict_item.get("media_bytes") is not None:
            # truncate source path since the media is embeded in arrow
            path = dict_item.get("media_path")
            if path is not None and source_path is not None:
                dict_item["media_path"] = path.replace(source_path, "")

        return dict_item
