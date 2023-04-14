# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import copy
import errno
import os
from collections.abc import Iterator
from typing import Dict, List, Optional, Tuple, TypeVar, Union, overload

import numpy as np
import pyarrow as pa

PathLike = TypeVar("PathLike", str, os.PathLike, None)


# TODO: consider replacing with `bisect` library
def _binary_search(arr, x, low=None, high=None, fn=None):
    low = 0 if low is None else low
    high = len(arr) - 1 if high is None else high
    if fn is None:

        def fn(arr, x):
            return arr[x]

    while low <= high:
        mid = (high + low) // 2
        val = fn(arr, mid)
        if val > x:
            high = mid - 1
        else:
            low = mid + 1
    return high


class ArrowDatasetIterator(Iterator):
    def __init__(self, dataset: "ArrowDataset"):
        self.dataset = dataset
        self.index = 0

    def __next__(self) -> Dict:
        if self.index >= len(self.dataset):
            raise StopIteration
        item = self.dataset[self.index]
        self.index += 1
        return item


class ArrowDataset:
    def __init__(
        self,
        files: Union[PathLike, List[PathLike]],
        keep_in_memory: bool = False,
    ):
        assert files
        if not isinstance(files, (list, tuple)):
            files = [files]

        self._files = files
        self._keep_in_memory = keep_in_memory

        self.unload_arrows()
        self.load_arrows()

    @property
    def table(self) -> pa.Table:
        self.load_arrows()
        return self.__table

    @property
    def column_names(self) -> List[str]:
        return self.table.column_names

    @property
    def num_columns(self) -> int:
        return self.table.num_columns

    @property
    def num_rows(self) -> int:
        return self.table.num_rows

    @property
    def shape(self) -> Tuple[int]:
        return self.table.shape

    @property
    def metadata(self) -> Optional[Dict[bytes, bytes]]:
        return self.table.schema.metadata

    def __deepcopy__(self, memo) -> "ArrowDataset":
        args = [self._files, self._keep_in_memory]
        copied_args = copy.deepcopy(args, memo)
        return self.__class__(*copied_args)

    def __len__(self) -> int:
        return self.table.num_rows

    def __iter__(self):
        return ArrowDatasetIterator(self)

    def unload_arrows(self) -> None:
        self.__table: pa.Table = None
        self.__batches: List[pa.RecordBatch] = []
        self.__offsets: List[int] = []

    def load_arrows(self) -> None:
        if self.__table is not None:
            return

        table = None
        for file in self._files:
            if not os.path.exists(file):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)

            if self._keep_in_memory:
                stream = pa.input_stream(file)
            else:
                stream = pa.memory_map(file)
            opened_stream = pa.ipc.open_stream(stream)
            table_ = opened_stream.read_all()
            if table:
                table = pa.concat_tables((table, table_))
            else:
                table = table_
        self._update_table(table)

    def _update_table(self, table: pa.Table) -> None:
        self.__table = table
        self.__batches = self.__table.to_batches()
        self.__offsets = np.cumsum([0] + [len(i) for i in self.__batches]).tolist()

    def flatten(self) -> "ArrowDataset":
        dataset = ArrowDataset(self._files, self._keep_in_memory)
        dataset._update_table(dataset.table.flatten())
        return dataset

    def get_batches(self, offset: int, length: int) -> List[pa.RecordBatch]:
        assert length > 0

        s_idx = _binary_search(self.__offsets, offset)
        e_idx = _binary_search(self.__offsets, offset + length - 1)

        batches = self.__batches[s_idx : e_idx + 1]
        if self.__offsets[s_idx + 1] >= offset + length:
            batches[0] = batches[0].slice(offset - self.__offsets[s_idx], length)
        else:
            batches[0] = batches[0].slice(offset - self.__offsets[s_idx])
            batches[-1] = batches[-1].slice(0, offset + length - self.__offsets[e_idx])

        return batches

    def slice(self, offset: int, length: int) -> List[Dict]:
        batches = self.get_batches(offset, length)
        items = pa.Table.from_batches(batches).to_pylist()
        return items

    @overload
    def __getitem__(self, key: str) -> "ArrowDataset":
        ...

    @overload
    def __getitem__(self, key: int) -> Dict:
        ...

    @overload
    def __getitem__(self, key: Union[range, slice]) -> List[Dict]:
        ...

    def __getitem__(
        self, key: Union[int, range, slice, str]
    ) -> Union[Dict, List[Dict], "ArrowDataset"]:
        if isinstance(key, str):
            if key not in self.column_names:
                raise KeyError(key)
            dataset = ArrowDataset(self._files, self._keep_in_memory)
            dataset._update_table(
                self.table.drop([name for name in self.column_names if name != key])
            )
            return dataset
        elif isinstance(key, slice):
            key = range(*key.indices(len(self)))

        if isinstance(key, range):
            if key.step == 1 and key.stop >= key.start:
                return self.slice(key.start, key.stop - key.start)
            return [
                self.slice(key_, 1)[0] if key_ >= 0 else self.slice(key_ + len(self), 1)[0]
                for key_ in key
            ]
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            return self.slice(key, 1)[0]

        raise KeyError(key)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_rows={len(self)}, "
            f"columns={self.column_names}, "
            f"keep_in_memory={self._keep_in_memory}"
            ")"
        )
