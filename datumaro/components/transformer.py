# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Optional

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.extractor import DatasetItem, IExtractor, _ExtractorBase
from datumaro.util import is_method_redefined


class Transform(_ExtractorBase, CliPlugin):
    """
    A base class for dataset transformations that change dataset items
    or their annotations.
    """

    @staticmethod
    def wrap_item(item, **kwargs):
        return item.wrap(**kwargs)

    def __init__(self, extractor: IExtractor):
        super().__init__()

        self._extractor = extractor

    def categories(self):
        return self._extractor.categories()

    def subsets(self):
        if self._subsets is None:
            self._subsets = set(self._extractor.subsets())
        return super().subsets()

    def __len__(self):
        assert self._length in {None, "parent"} or isinstance(self._length, int)
        if (
            self._length is None
            and not is_method_redefined("__iter__", Transform, self)
            or self._length == "parent"
        ):
            self._length = len(self._extractor)
        return super().__len__()

    def media_type(self):
        return self._extractor.media_type()


class ItemTransform(Transform):
    def transform_item(self, item: DatasetItem) -> Optional[DatasetItem]:
        """
        Returns a modified copy of the input item.

        Avoid changing and returning the input item, because it can lead to
        unexpected problems. Use wrap_item() or item.wrap() to simplify copying.
        """

        raise NotImplementedError()

    def __iter__(self):
        for item in self._extractor:
            item = self.transform_item(item)
            if item is not None:
                yield item
