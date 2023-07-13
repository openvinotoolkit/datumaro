# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
from typing import Sequence

from datumaro.components.contexts.importer import _ImportFail
from datumaro.components.dataset_base import SubsetBase
from datumaro.components.merge.base import Merger
from datumaro.components.merge.extractor_merger import ExtractorMerger, check_identicalness
from datumaro.plugins.data_formats.coco.base import _CocoBase
from datumaro.plugins.data_formats.coco.format import CocoTask
from datumaro.plugins.data_formats.coco.page_mapper import MergedCOCOPageMapper


class COCOTaskMergedBase(_CocoBase):
    def __init__(
        self,
        sources: Sequence[_CocoBase],
        subset: str,
    ):
        SubsetBase.__init__(self, subset=subset, ctx=None)
        page_mappers = []
        for s in sources:
            if not s.is_stream:
                raise NotImplementedError("For now, support is_stream=True only.")
            page_mappers.append(s.page_mapper)

        self._page_mapper = MergedCOCOPageMapper.create(page_mappers)
        self._length = None
        self._path = ",".join(s.path for s in sources)
        self._rootpath = check_identicalness([s.rootpath for s in sources])
        self._images_dir = check_identicalness([s.images_dir for s in sources])
        self._task = CocoTask.null

        for s in sources:
            self._task |= s.task

        self._label_map = check_identicalness(
            [s.label_map for s in sources if len(s.label_map) > 0],
            raise_error_on_empty=False,
        )
        self._merge_instance_polygons = check_identicalness(
            [s.merge_instance_polygons for s in sources]
        )
        self._mask_dir = check_identicalness(
            [s.mask_dir for s in sources if s.mask_dir is not None],
            raise_error_on_empty=False,
        )

    @property
    def is_stream(self) -> bool:
        return True


class COCOExtractorMerger(ExtractorMerger):
    def __init__(self, sources: Sequence[_CocoBase]):
        if len(sources) == 0:
            raise _ImportFail("It should not be empty.")

        self._infos = check_identicalness([s.infos() for s in sources])
        self._categories = Merger.merge_categories([s.categories() for s in sources])
        self._media_type = check_identicalness([s.media_type() for s in sources])
        self._is_stream = check_identicalness([s.is_stream for s in sources])

        grouped_by_subset = defaultdict(list)

        for s in sources:
            grouped_by_subset[s.subset] += [s]

        self._subsets = {
            subset: COCOTaskMergedBase(sources, subset)
            for subset, sources in grouped_by_subset.items()
        }
