# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import struct

import pyarrow as pa

from datumaro.components.dataset_base import SubsetBase
from datumaro.components.errors import MediaTypeError
from datumaro.components.media import MediaType
from datumaro.components.merge import get_merger
from datumaro.plugins.data_formats.datumaro.base import DatumaroBase
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import DictMapper

from .arrow_dataset import ArrowDataset
from .mapper.dataset_item import DatasetItemMapper


class ArrowBase(SubsetBase):
    def __init__(self, path, ctx, subset, additional_paths=[]):
        super().__init__(subset=subset, ctx=ctx)

        self._paths = [path] + additional_paths

        self._load()

    def _load(self):
        infos = []
        categories = []
        media_types = set()

        for path in self._paths:
            with pa.ipc.open_stream(path) as reader:
                schema = reader.schema

                _infos, _ = DictMapper.backward(schema.metadata.get(b"infos", b"\x00\x00\x00\x00"))
                infos.append(DatumaroBase._load_infos({"infos": _infos}))

                _categories, _ = DictMapper.backward(
                    schema.metadata.get(b"categories", b"\x00\x00\x00\x00")
                )
                categories.append(DatumaroBase._load_categories({"categories": _categories}))

                (media_type,) = struct.unpack(
                    "<I", schema.metadata.get(b"media_type", b"\x00\x00\x00\x00")
                )
                media_types.add(MediaType(media_type).media)

                dataset = ArrowDataset(path)
                dataset = dataset.flatten()

                for i in self._ctx.progress_reporter.iter(
                    range(len(dataset)), desc=f"Reading arrow from '{osp.basename(path)}'"
                ):
                    batches = dataset.get_batches(i, 1)
                    self._items.extend(DatasetItemMapper.backward_from_batches(batches))

        if len(media_types) > 1:
            raise MediaTypeError("Datasets have different media types")
        merger = get_merger("exact")
        self._infos = merger.merge_infos(infos)
        self._categories = merger.merge_categories(categories)
        self._media_type = list(media_types)[0]
