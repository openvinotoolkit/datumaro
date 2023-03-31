# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

import pyarrow as pa

from datumaro.components.dataset_base import SubsetBase
from datumaro.plugins.data_formats.datumaro.base import DatumaroBase
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import DictMapper

from .arrow_dataset import ArrowDataset
from .mapper.dataset_item import DatasetItemMapper


class ArrowBase(SubsetBase):
    def __init__(self, path):
        self._path = path
        self._schema = None
        self._infos = None
        self._categories = None

        super().__init__(subset=osp.splitext(osp.basename(path))[0])

        self._load()

    def _load(self):
        with pa.ipc.open_stream(self._path) as reader:
            self._schema = reader.schema

            infos, _ = DictMapper.backward(self._schema.metadata.get(b"infos", b"\x00\x00\x00\x00"))
            self._infos = DatumaroBase._load_infos({"infos": infos})

            categories, _ = DictMapper.backward(
                self._schema.metadata.get(b"categories", b"\x00\x00\x00\x00")
            )
            self._categories = DatumaroBase._load_categories({"categories": categories})

            dataset = ArrowDataset(self._path)
            dataset = dataset.flatten()

            for i in range(len(dataset)):
                batches = dataset.get_batches(i, 1)
                self._items.extend(DatasetItemMapper.backward_from_batches(batches))
