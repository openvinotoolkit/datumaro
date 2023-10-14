# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
from dataclasses import dataclass
import os.path as osp
import struct
from typing import Any, Dict, Iterator, List, Optional, Type

import pyarrow as pa
from datumaro.components.annotation import AnnotationType, Categories

from datumaro.components.dataset_base import DatasetBase, DatasetItem, IDataset, SubsetBase
from datumaro.components.errors import MediaTypeError
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image, MediaElement, MediaType
from datumaro.components.merge import get_merger
from datumaro.components.merge.extractor_merger import check_identicalness
from datumaro.plugins.data_formats.arrow.format import DatumaroArrow
from datumaro.plugins.data_formats.datumaro.base import JsonReader
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import DictMapper
from datumaro.util.definitions import DEFAULT_SUBSET_NAME
import weakref

from .mapper.dataset_item import DatasetItemMapper


class ArrowSubsetBase(SubsetBase):
    """
    A base class for simple, single-subset extractors.
    Should be used by default for user-defined extractors.
    """

    def __init__(
        self,
        lookup: Dict[str, int],
        table: pa.Table,
        infos: Dict[str, Any],
        categories: Dict[AnnotationType, Categories],
        subset: str,
        media_type: Type[MediaElement] = Image,
    ):
        super().__init__(length=len(lookup), subset=subset, media_type=media_type, ctx=None)

        self._lookup = lookup
        self._table = table
        self._infos = infos
        self._categories = categories

    def __iter__(self) -> Iterator[DatasetItem]:
        for table_idx in self._lookup.values():
            yield DatasetItemMapper.backward(self._table[table_idx], self._table)

    def __len__(self):
        return len(self._lookup)

    def get(self, item_id: str, subset: Optional[str] = None):
        if subset != self._subset:
            return None

        try:
            table_idx = self._lookup[item_id]
            return DatasetItemMapper.backward(table_idx, self._table)
        except KeyError:
            return None


@dataclass(frozen=True)
class Metadata:
    infos: Dict
    categories: Dict
    media_type: Type[MediaElement]


class ArrowBase(DatasetBase):
    def __init__(
        self,
        root_path: str,
        *,
        file_paths: List[str],
        ctx: Optional[ImportContext] = None,
    ):
        self._root_path = root_path
        tables = [pa.ipc.open_file(pa.memory_map(path, "r")).read_all() for path in file_paths]
        metadatas = [self._load_schema_metadata(table) for table in tables]

        table = pa.concat_tables(tables)
        subsets = table.column(DatumaroArrow.SUBSET_FIELD).unique().to_pylist()
        media_type = check_identicalness([metadata.media_type for metadata in metadatas])

        super().__init__(length=len(table), subsets=subsets, media_type=media_type, ctx=ctx)

        self._infos = check_identicalness([metadata.infos for metadata in metadatas])
        self._categories = check_identicalness([metadata.categories for metadata in metadatas])

        self._init_cache(table)

    @staticmethod
    def _load_schema_metadata(table: pa.Table) -> Metadata:
        schema = table.schema

        _infos, _ = DictMapper.backward(schema.metadata.get(b"infos", b"\x00\x00\x00\x00"))
        infos = JsonReader._load_infos({"infos": _infos})

        _categories, _ = DictMapper.backward(
            schema.metadata.get(b"categories", b"\x00\x00\x00\x00")
        )
        categories = JsonReader._load_categories({"categories": _categories})

        (media_type,) = struct.unpack("<I", schema.metadata.get(b"media_type", b"\x00\x00\x00\x00"))
        media_type = MediaType(media_type).media

        return Metadata(infos=infos, categories=categories, media_type=media_type)

    def infos(self):
        return self._infos

    def categories(self):
        return self._categories

    def __iter__(self) -> Iterator[DatasetItem]:
        for idx in range(len(self)):
            yield DatasetItemMapper.backward(idx, self._table)

    def _init_cache(self, table: pa.Table):
        self._lookup = defaultdict(dict)

        for idx, (item_id, item_subset) in enumerate(
            zip(
                table.column(DatumaroArrow.ID_FIELD),
                table.column(DatumaroArrow.SUBSET_FIELD),
            )
        ):
            self._lookup[item_subset.as_py()][item_id.as_py()] = idx

        self._subsets = {
            subset: ArrowSubsetBase(
                lookup=lookup,
                table=table,
                infos=self._infos,
                categories=self._categories,
                subset=self._subsets,
                media_type=self._media_type,
            )
            for subset, lookup in self._lookup.items()
        }
        self._table = table

    def get(self, item_id: str, subset: Optional[str] = None) -> Optional[DatasetItem]:
        subset = subset or DEFAULT_SUBSET_NAME

        try:
            table_idx = self._lookup[subset][item_id]
            return DatasetItemMapper.backward(self._table[table_idx], self._table)
        except KeyError:
            return None

    @property
    def lookup(self) -> Dict[str, Dict[str, int]]:
        return self._lookup

    def subsets(self) -> Dict[str, IDataset]:
        return self._subsets

    def get_subset(self, name: str) -> IDataset:
        return self._subsets[name]
