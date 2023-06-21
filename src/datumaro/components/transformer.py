# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT
from typing import Generator, List, Optional

import numpy as np

from datumaro.components.annotation import Annotation, AnnotationType, LabelCategories
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DatasetBase, DatasetItem, IDataset
from datumaro.components.launcher import Launcher
from datumaro.util import is_method_redefined, take_by


class Transform(DatasetBase, CliPlugin):
    """
    A base class for dataset transformations that change dataset items
    or their annotations.
    """

    @staticmethod
    def wrap_item(item, **kwargs):
        return item.wrap(**kwargs)

    def __init__(self, extractor: IDataset):
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


class ModelTransform(Transform):
    def __init__(
        self,
        extractor: IDataset,
        launcher: Launcher,
        batch_size: int = 1,
        append_annotation: bool = False,
    ):
        super().__init__(extractor)
        self._launcher = launcher
        self._batch_size = batch_size
        self._append_annotation = append_annotation

    def __iter__(self) -> Generator[DatasetItem, None, None]:
        for batch in take_by(self._extractor, self._batch_size):
            inference = self._launcher.launch(
                [item for item in batch if self._launcher.type_check(item)]
            )

            for item in self._yield_item(batch, inference):
                yield item

    def _yield_item(
        self, batch: List[DatasetItem], inference: List[List[Annotation]]
    ) -> Generator[DatasetItem, None, None]:
        for item, annotations in zip(batch, inference):
            self._check_annotations(annotations)
            if self._append_annotation:
                annotations = item.annotations + annotations
            yield self.wrap_item(item, annotations=annotations)

    def get_subset(self, name):
        subset = self._extractor.get_subset(name)
        return __class__(subset, self._launcher, self._batch_size)

    def infos(self):
        launcher_override = self._launcher.infos()
        if launcher_override is not None:
            return launcher_override
        return self._extractor.infos()

    def categories(self):
        launcher_override = self._launcher.categories()
        if launcher_override is not None:
            return launcher_override
        return self._extractor.categories()

    def transform_item(self, item):
        inputs = np.expand_dims(item.media, axis=0)
        annotations = self._launcher.launch(inputs)[0]
        return self.wrap_item(item, annotations=annotations)

    def _check_annotations(self, annotations: List[Annotation]):
        labels_count = len(self.categories().get(AnnotationType.label, LabelCategories()).items)

        for ann in annotations:
            label = getattr(ann, "label", None)
            if label is None:
                continue

            if label not in range(labels_count):
                raise Exception(
                    "Annotation has unexpected label id %s, "
                    "while there is only %s defined labels." % (label, labels_count)
                )
