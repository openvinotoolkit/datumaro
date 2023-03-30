# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.dataset import DatasetItemStorage
from datumaro.components.dataset_base import IDataset
from datumaro.components.merge import Merger

__all__ = ["UnionMerge"]


class UnionMerge(Merger):
    """
    Merges several datasets using the "simple" algorithm:
        - items are matched by (id, subset) pairs
        - matching items share the media info available:
            - nothing + nothing = nothing
            - nothing + something = something
            - something A + something B = something (A + B)
        - annotations are matched by value and shared
        - in case of conflicts, throws an error
    """

    def __init__(self, **options):
        super().__init__(**options)
        self._matching_table = {}

    def merge(self, *sources: IDataset) -> DatasetItemStorage:
        items = DatasetItemStorage()
        for source_idx, source in enumerate(sources):
            for item in source:
                if self._matching_table.get(source_idx, None):
                    for ann in item.annotations:
                        ann.label = self._matching_table[source_idx][ann.label]
                items.put(item)
        return items

    def merge_categories(self, sources):
        dst_categories = {}

        label_cat = self._merge_label_categories(sources)
        if label_cat is None:
            label_cat = LabelCategories()
        dst_categories[AnnotationType.label] = label_cat

        return dst_categories

    def _merge_label_categories(self, sources):
        dst_cat = LabelCategories()
        dst_indices = {}
        dst_labels = []

        for src_id, src_categories in enumerate(sources):
            src_cat = src_categories.get(AnnotationType.label)
            if src_cat is None:
                continue

            for src_label in src_cat.items:
                if src_label.name not in dst_labels:
                    dst_cat.add(src_label.name, src_label.parent, src_label.attributes)
                    dst_labels.append(src_label.name)

                    if src_cat._indices[src_label.name] in list(dst_indices.values()):
                        dst_indices[src_label.name] = max(dst_indices.values()) + 1
                        if self._matching_table.get(src_id, None):
                            self._matching_table[src_id].update(
                                {src_cat._indices[src_label.name]: max(dst_indices.values())}
                            )
                        else:
                            self._matching_table[src_id] = {
                                src_cat._indices[src_label.name]: max(dst_indices.values())
                            }
                    else:
                        dst_indices[src_label.name] = src_cat._indices[src_label.name]
                else:
                    if self._matching_table.get(src_id, None):
                        self._matching_table[src_id].update(
                            {src_cat._indices[src_label.name]: dst_indices[src_label.name]}
                        )
                    else:
                        self._matching_table[src_id] = {
                            src_cat._indices[src_label.name]: dst_indices[src_label.name]
                        }

        return dst_cat
