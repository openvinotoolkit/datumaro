# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Any, Dict, List, Optional

import pyarrow as pa

from datumaro.components.dataset_base import DatasetItem
from datumaro.plugins.data_formats.datumaro_binary.mapper.annotation import AnnotationListMapper
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import DictMapper, Mapper

from .media import MediaMapper
from .utils import pa_batches_decoder


class DatasetItemMapper(Mapper):
    @staticmethod
    def forward(obj: DatasetItem, **options) -> Dict[str, Any]:
        return {
            "id": obj.id,
            "subset": obj.subset,
            "media": MediaMapper.forward(obj.media, **options.get("media", {})),
            "annotations": AnnotationListMapper.forward(obj.annotations),
            "attributes": DictMapper.forward(obj.attributes),
        }

    @staticmethod
    def backward(obj: Dict[str, Any]) -> DatasetItem:
        return DatasetItem(
            id=obj["id"],
            subset=obj["subset"],
            media=MediaMapper.backward(obj["media"]),
            annotations=AnnotationListMapper.backward(obj["annotations"])[0],
            attributes=DictMapper.backward(obj["attributes"])[0],
        )

    @staticmethod
    def backward_from_batches(
        batches: List[pa.lib.RecordBatch],
        parent: Optional[str] = None,
    ) -> List[DatasetItem]:
        ids = pa_batches_decoder(batches, f"{parent}.id" if parent else "id")
        subsets = pa_batches_decoder(batches, f"{parent}.subset" if parent else "subset")
        medias = MediaMapper.backward_from_batches(
            batches, f"{parent}.media" if parent else "media"
        )
        annotations_ = pa_batches_decoder(
            batches, f"{parent}.annotations" if parent else "annotations"
        )
        annotations_ = [
            AnnotationListMapper.backward(annotations)[0] for annotations in annotations_
        ]
        attributes_ = pa_batches_decoder(
            batches, f"{parent}.attributes" if parent else "attributes"
        )
        attributes_ = [DictMapper.backward(attributes)[0] for attributes in attributes_]

        items = []
        for id, subset, media, annotations, attributes in zip(
            ids, subsets, medias, annotations_, attributes_
        ):
            items.append(
                DatasetItem(
                    id=id,
                    subset=subset,
                    media=media,
                    annotations=annotations,
                    attributes=attributes,
                )
            )
        return items
