# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, TypeVar

from datumaro.components.contexts.importer import _ImportFail
from datumaro.components.dataset_base import (
    CategoriesInfo,
    DatasetBase,
    DatasetInfo,
    DatasetItem,
    SubsetBase,
)
from datumaro.components.task import TaskAnnotationMapping

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

        task_annotation_mapping = TaskAnnotationMapping()
        ann_types = set()
        for source in sources:
            for ann_type in task_annotation_mapping[source.task_type()]:
                ann_types.add(ann_type)
        self._task_type = task_annotation_mapping.get_task(ann_types)

        self._is_stream = check_identicalness([s.is_stream for s in sources])

        self._subsets: Dict[str, List[SubsetBase]] = defaultdict(list)
        for source in sources:
            self._subsets[source.subset] += [source]

    def infos(self) -> DatasetInfo:
        return self._infos

    def categories(self) -> CategoriesInfo:
        return self._categories

    def __iter__(self) -> Iterator[DatasetItem]:
        for sources in self._subsets.values():
            for source in sources:
                yield from source

    def __len__(self) -> int:
        return sum(len(source) for sources in self._subsets.values() for source in sources)

    def get(self, id: str, subset: Optional[str] = None) -> Optional[DatasetItem]:
        if subset is not None and (sources := self._subsets.get(subset, [])):
            for source in sources:
                if item := source.get(id, subset):
                    return item

        for sources in self._subsets.values():
            for source in sources:
                if item := source.get(id=id, subset=source.subset):
                    return item

        return None

    @property
    def is_stream(self) -> bool:
        return self._is_stream
