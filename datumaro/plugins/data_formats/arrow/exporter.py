# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import datetime
import os
import platform
import re
import struct
import tempfile
from copy import deepcopy
from functools import partial
from multiprocessing.pool import ApplyResult, Pool
from shutil import move, rmtree
from typing import Any, Callable, Dict, Optional, Union

import pyarrow as pa
import pytz

from datumaro.components.crypter import NULL_CRYPTER
from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetItem, IDataset
from datumaro.components.errors import DatumaroError
from datumaro.components.exporter import ExportContext, ExportContextComponent, Exporter
from datumaro.plugins.data_formats.datumaro.exporter import _SubsetWriter as __SubsetWriter
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import DictMapper
from datumaro.util.file_utils import to_bytes

from .format import DatumaroArrow
from .mapper.dataset_item import DatasetItemMapper
from .mapper.media import ImageMapper


class PathNormalizer:
    NORMALIZER = {r":[^/\\]": r"꞉"}
    UNNORMALIZER = {r"꞉": r":"}

    @classmethod
    def normalize(cls, path: str):
        for s, t in cls.NORMALIZER.items():
            path = re.sub(s, t, path)
        return path

    @classmethod
    def unnormalize(cls, path: str):
        for s, t in cls.UNNORMALIZER.items():
            path = re.sub(s, t, path)
        return path


class _SubsetWriter(__SubsetWriter):
    def __init__(
        self,
        context: Exporter,
        export_context: ExportContextComponent,
        subset: str,
        ctx: ExportContext,
        max_chunk_size: int = 1000,
        num_shards: int = 1,
        max_shard_size: Optional[int] = None,
    ):
        super().__init__(context, "", export_context)
        self._schema = deepcopy(DatumaroArrow.SCHEMA)
        self._subset = subset
        self._writers = []
        self._fnames = []
        self._max_chunk_size = max_chunk_size
        self._num_shards = num_shards
        self._max_shard_size = max_shard_size
        self._ctx = ctx

        self._data = {
            "items": [],
            "infos": {},
            "categories": {},
            "media_type": None,
            "built_time": str(datetime.datetime.now(pytz.utc)),
            "source_path": self.export_context.source_path,
            "encoding_scheme": self.export_context._image_ext,
            "version": str(DatumaroArrow.VERSION),
            "signature": DatumaroArrow.SIGNATURE,
        }

    def add_infos(self, infos):
        if self._writers:
            raise ValueError("Writer has been initialized.")
        super().add_infos(infos)
        self._data["infos"] = DictMapper.forward(self.infos)

    def add_categories(self, categories):
        if self._writers:
            raise ValueError("Writer has been initialized.")
        super().add_categories(categories)
        self._data["categories"] = DictMapper.forward(self.categories)

    def add_media_type(self, media_type):
        if self._writers:
            raise ValueError("Writer has been initialized.")
        self._data["media_type"] = struct.pack("<I", int(media_type))

    def init_schema(self):
        self._schema = self._schema.with_metadata(
            {k: v for k, v in self._data.items() if k != "items"}
        )

    def _init_writer(self, idx: int):
        # TODO:
        # `arrow_writer.close()` does not take any effect. It seems a bug in pyarrow
        # In linux-like system it is fine to rename a file opend but
        # it is not in windows.
        # As a workaround, we do not use a template in windows
        if platform.system() != "Windows":
            f_name = os.path.join(
                self.export_context.save_dir,
                self._subset + "-{idx" + str(idx) + ":0{width}d}-of-{total:0{width}d}.arrow",
            )
            f_name = PathNormalizer.normalize(f_name)
        else:
            f_name = os.path.join(
                self.export_context.save_dir, self._subset + f"-{idx:03d}-of-{0:03d}.arrow"
            )
        return pa.RecordBatchStreamWriter(f_name, self._schema), f_name

    def add_item(self, item: DatasetItem, pool: Optional[Pool] = None):
        if pool is not None:
            self.items.append(
                pool.apply_async(
                    self.add_item_impl,
                    (
                        item,
                        self.export_context,
                    ),
                )
            )
        else:
            self.items.append(partial(self.add_item_impl, item, self.export_context))

    @staticmethod
    def add_item_impl(
        item: DatasetItem,
        context: ExportContextComponent,
    ) -> Dict[str, Any]:
        item = DatasetItemMapper.forward(item, media={"encoder": context._image_ext})

        if item["media"].get("bytes", None) is not None:
            # truncate source path since the media is embeded in arrow
            path = item["media"].get("path")
            if path is not None:
                item["media"]["path"] = path.replace(context.source_path, "")
        return item

    def _write(self, batch, max_chunk_size):
        if max_chunk_size == 0:
            max_chunk_size = self._max_chunk_size
        pa_table = pa.Table.from_arrays(batch, schema=self._schema)
        idx = getattr(self, "__writer_idx", 0)

        if self._max_shard_size is not None:
            nbytes = pa_table.nbytes
            cur_nbytes = getattr(self, "__writer_nbytes", 0)
            free_nbytes = self._max_shard_size - cur_nbytes
            if free_nbytes < nbytes:
                if cur_nbytes == 0:
                    raise DatumaroError(
                        "'max_chunk_size' exceeded 'max_shard_size'. "
                        "Please consider increasing 'max_shard_size' or deceasing 'max_chunk_size'."
                    )
                self._writers[idx].close()
                idx += 1
                setattr(self, "__writer_idx", idx)
                setattr(self, "__writer_nbytes", nbytes)
            else:
                setattr(self, "__writer_nbytes", cur_nbytes + nbytes)
        else:
            setattr(self, "__writer_idx", (idx + 1) % self._num_shards)

        if len(self._writers) <= idx:
            writer, fname = self._init_writer(idx)
            self._writers.append(writer)
            self._fnames.append(fname)
        self._writers[idx].write_table(pa_table, max_chunk_size)

    def write(self, max_chunk_size: Optional[int] = None, pool: Optional[Pool] = None):
        if max_chunk_size is None:
            max_chunk_size = self._max_chunk_size

        if len(self.items) < max_chunk_size:
            return

        batch = [[] for _ in self._schema.names]
        for item in self._ctx.progress_reporter.iter(
            self.items, desc=f"Building arrow for {self._subset}"
        ):
            if isinstance(item, ApplyResult):
                item = item.get(timeout=DatumaroArrow.MP_TIMEOUT)
            if isinstance(item, partial):
                item = item()
            for j, name in enumerate(self._schema.names):
                batch[j].append(item[name])

            if len(batch[0]) >= max_chunk_size:
                self._write(batch, max_chunk_size)
                batch = [[] for _ in self._schema.names]
        if len(batch[0]) > 0:
            self._write(batch, max_chunk_size)

        self._data["items"] = []

    def done(self):
        total = len(self._fnames)
        width = len(str(total))
        for idx, (fname, writer) in enumerate(zip(self._fnames, self._writers)):
            writer.close()
            if platform.system() != "Windows":
                template = PathNormalizer.unnormalize(fname)
                placeholders = {"width": width, "total": total, f"idx{idx}": idx}
                new_fname = template.format(**placeholders)
                if os.path.exists(new_fname):
                    os.remove(new_fname)
                os.rename(fname, new_fname)


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
            help="The maximum chunk size. (default: %(default)s)",
        )

        parser.add_argument(
            "--num-shards",
            type=int,
            default=1,
            help="The number of shards to export. "
            "'--num-shards' and '--max-shard-size' are  mutually exclusive. "
            "(default: %(default)s)",
        )

        parser.add_argument(
            "--max-shard-size",
            type=str,
            default=None,
            help="The maximum size of each shard. "
            "(e.g. 7KB = 7 * 2^10, 3MB = 3 * 2^20, and 2GB = 2 * 2^30). "
            "'--num-shards' and '--max-shard-size' are  mutually exclusive. "
            "(default: %(default)s)",
        )

        parser.add_argument(
            "--num-workers",
            type=int,
            default=0,
            help="The number of multi-processing workers for export. "
            "If num_workers = 0, do not use multiprocessing. (default: %(default)s)",
        )

        return parser

    def create_writer(self, subset: str, ctx: ExportContext) -> _SubsetWriter:
        export_context = ExportContextComponent(
            save_dir=self._save_dir,
            save_media=self._save_media,
            images_dir="",
            pcd_dir="",
            crypter=NULL_CRYPTER,
            image_ext=self._image_ext,
            default_image_ext=self._default_image_ext,
            source_path=os.path.abspath(self._extractor._source_path)
            if getattr(self._extractor, "_source_path")
            else None,
        )

        return _SubsetWriter(
            context=self,
            subset=subset,
            export_context=export_context,
            num_shards=self._num_shards,
            max_shard_size=self._max_shard_size,
            max_chunk_size=self._max_chunk_size,
            ctx=ctx,
        )

    def apply(self, *args, **kwargs):
        if self._num_workers == 0:
            return self._apply()

        with Pool(processes=self._num_workers) as pool:
            return self._apply(pool)

    def _apply(self, pool: Optional[Pool] = None):
        os.makedirs(self._save_dir, exist_ok=True)

        if self._split_by_subsets:
            writers = {
                subset_name: self.create_writer(subset_name, self._ctx)
                for subset_name, subset in self._extractor.subsets().items()
                if len(subset)
            }
        else:
            writers = {DEFAULT_SUBSET_NAME: self.create_writer(DEFAULT_SUBSET_NAME, self._ctx)}

        for writer in writers.values():
            writer.add_infos(self._extractor.infos())
            writer.add_categories(self._extractor.categories())
            writer.add_media_type(self._extractor.media_type()._type)
            writer.init_schema()

        for subset_name, subset in self._extractor.subsets().items():
            writer = (
                writers[subset_name] if self._split_by_subsets else writers[DEFAULT_SUBSET_NAME]
            )
            for item in subset:
                writer.add_item(item, pool=pool)

        for writer in writers.values():
            writer.write(0, pool=pool)
            writer.done()

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
        save_images=None,  # Deprecated
        save_media: Optional[bool] = None,
        image_ext: Optional[Union[str, Callable[[str], bytes]]] = None,
        default_image_ext: Optional[str] = None,
        save_dataset_meta: bool = False,
        ctx: Optional[ExportContext] = None,
        num_workers: int = 0,
        num_shards: int = 1,
        max_shard_size: Optional[int] = None,
        max_chunk_size: int = 1000,
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

        # TODO: Support a whole single file of arrow
        self._split_by_subsets = True

        if num_workers < 0:
            raise DatumaroError(
                f"num_workers should be non-negative but num_workers={num_workers}."
            )
        self._num_workers = num_workers

        if num_shards != 1 and max_shard_size is not None:
            raise DatumaroError(
                "Either one of 'num_shards' or 'max_shard_size' should be provided, not both."
            )

        if num_shards < 0:
            raise DatumaroError(f"num_shards should be non-negative but num_shards={num_shards}.")
        self._num_shards = num_shards
        self._max_shard_size = to_bytes(max_shard_size) if max_shard_size else max_shard_size

        if max_chunk_size < 0:
            raise DatumaroError(
                f"max_chunk_size should be non-negative but max_chunk_size={max_chunk_size}."
            )
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
