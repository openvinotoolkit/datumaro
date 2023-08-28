# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Optional, Sequence, TypeVar

from datumaro.components.contexts.importer import _ImportFail
from datumaro.components.dataset_base import DatasetBase, SubsetBase

T = TypeVar("T")


def check_identicalness(seq: Sequence[T], raise_error_on_empty: bool = True) -> Optional[T]:
    if len(seq) == 0 and raise_error_on_empty:
        raise _ImportFail("It should not be empty.")
    elif len(seq) == 0 and not raise_error_on_empty:
        return None

    if seq.count(seq[0]) != len(seq):
        raise _ImportFail("All items in the sequence should be identical.")

    return seq[0]


class ExtractorMerger(DatasetBase):
    """A simple class to merge single-subset extractors."""

    def __init__(
        self,
        sources: Sequence[SubsetBase],
    ):
        if len(sources) == 0:
            raise _ImportFail("It should not be empty.")

        self._infos = check_identicalness([s.infos() for s in sources])
        self._categories = check_identicalness([s.categories() for s in sources])
        self._media_type = check_identicalness([s.media_type() for s in sources])
        self._is_stream = check_identicalness([s.is_stream for s in sources])
        self._subsets = {s.subset: s for s in sources}

    def infos(self):
        return self._infos

    def categories(self):
        return self._categories

    def __iter__(self):
        for subset in self._subsets.values():
            yield from subset

    def __len__(self):
        return sum(len(subset) for subset in self._subsets.values())

    def get(self, id: str, subset: Optional[str] = None):
        if subset is None:
            for s in self._subsets.values():
                item = s.get(id)
                if item is not None:
                    return item

        s = self._subsets[subset]
        return s.get(id, subset)

    @property
    def is_stream(self) -> bool:
        return self._is_stream
