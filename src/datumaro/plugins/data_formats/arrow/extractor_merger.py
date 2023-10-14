# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
from typing import Any, Dict, Iterator, Optional, Sequence, Type
import weakref
from datumaro.components.annotation import AnnotationType, Categories, LabelCategories

from datumaro.components.contexts.importer import _ImportFail, ImportContext
from datumaro.components.dataset_base import DatasetItem, IDataset, SubsetBase
from datumaro.components.media import Image, MediaElement
from datumaro.components.merge.extractor_merger import ExtractorMerger, check_identicalness
from datumaro.plugins.data_formats.arrow.base import ArrowBase
from datumaro.util.definitions import DEFAULT_SUBSET_NAME


class ArrowSubsetBase(SubsetBase):
    """
    A base class for simple, single-subset extractors.
    Should be used by default for user-defined extractors.
    """

    def __init__(
        self,
        lookup: Dict[str, weakref.ReferenceType[ArrowBase]],
        infos: Dict[str, Any],
        categories: Dict[AnnotationType, Categories],
        subset: str,
        media_type: Type[MediaElement] = Image,
    ):
        super().__init__(length=len(lookup), subsets=self.subset, media_type=media_type, ctx=None)

        self._lookup = lookup
        self._infos = infos
        self._categories = categories

    def infos(self):
        return self._infos

    def categories(self):
        return self._categories

    def __iter__(self):
        for source in self._lookup.values():
            source.
        yield from self._items

    def __len__(self):
        return len(self._items)

    def get(self, item_id: str, subset: Optional[str] = None):
        if subset != self._subset:
            return None

        try:
            ref_source = self._lookup[item_id]
            if (source := ref_source()) is not None:
                return source.get(item_id, subset)
        except KeyError:
            return None
        return None

    @property
    def subset(self) -> str:
        """Subset name of this instance."""
        return self._subset


class ArrowBaseMerger(ExtractorMerger):
    __not_plugin__ = True

    def __init__(self, sources: Sequence[ArrowBase]):
        if len(sources) == 0:
            raise _ImportFail("It should not be empty.")

        self._infos = check_identicalness([s.infos() for s in sources])
        self._categories = check_identicalness([s.categories() for s in sources])
        self._media_type = check_identicalness([s.media_type() for s in sources])
        self._is_stream = check_identicalness([s.is_stream for s in sources])

        self._sources = sources

        self._lookup: Dict[str, Dict[str, weakref.ReferenceType[ArrowBase]]] = defaultdict(dict)

        for source in sources:
            for subset, item_to_idx_lookup in source.lookup.items():
                for item_id, table_idx in item_to_idx_lookup.items():
                    self._lookup[subset][item_id] = (weakref.ref(source), table_idx)

    def __iter__(self) -> Iterator[DatasetItem]:
        for source in self._sources:
            yield from source

    def __len__(self) -> int:
        return sum(len(source) for source in self._sources)

    def get(self, item_id: str, subset: Optional[str] = None):
        subset = subset or DEFAULT_SUBSET_NAME

        try:
            ref_source = self._lookup[subset][item_id]
            if (source := ref_source()) is not None:
                return source.get(item_id, subset)
        except KeyError:
            return None

    def subsets(self) -> Dict[str, IDataset]:
        if self._subsets is None:
            self._init_cache()
        return {name or DEFAULT_SUBSET_NAME: self.get_subset(name) for name in self._subsets}

    def get_subset(self, name: str) -> IDataset:
        if self._subsets is None:
            self._init_cache()
        if name in self._subsets:
            if len(self._subsets) == 1:
                return self

            subset = self.select(lambda item: item.subset == name)
            subset._subsets = [name]
            return subset
        else:
            raise KeyError(
                "Unknown subset '%s', available subsets: %s" % (name, set(self._subsets))
            )
