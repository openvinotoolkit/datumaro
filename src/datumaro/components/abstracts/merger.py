# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Type

from datumaro.components.dataset_base import IDataset
from datumaro.components.dataset_item_storage import (
    DatasetItemStorage,
    DatasetItemStorageDatasetView,
)
from datumaro.components.media import MediaElement

__all__ = ["IMerger"]


class IMerger(ABC):
    @abstractmethod
    def merge_infos(self, sources: Sequence[IDataset]) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def merge_categories(self, sources: Sequence[IDataset]) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def merge_media_types(self, sources: Sequence[IDataset]) -> Optional[Type[MediaElement]]:
        raise NotImplementedError

    @abstractmethod
    def merge(self, sources: Sequence[IDataset]) -> DatasetItemStorage:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *datasets: IDataset) -> DatasetItemStorageDatasetView:
        raise NotImplementedError
