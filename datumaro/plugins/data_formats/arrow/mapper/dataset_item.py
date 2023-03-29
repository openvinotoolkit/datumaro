# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import struct
from typing import Any, Dict, List

import pyarrow as pa

from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image, MediaType
from datumaro.plugins.data_formats.datumaro_binary.mapper.annotation import AnnotationListMapper
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import DictMapper, Mapper

from .media import ImageFileMapper, MediaMapper


DEFAULT_DECODE_KEYS = {
    "attributes": lambda obj: DictMapper.backward(obj)[0],
    "annotations": lambda obj: AnnotationListMapper.backward(obj)[0],
    "media.attributes": lambda obj: DictMapper.backward(obj)[0],
}


def arrow_decoder(batches, column=None):
    if not isinstance(batches, list):
        batches = [batches]
    table = pa.Table.from_batches(batches)
    if column:
        data = table.column(column).to_pylist()
        if column in DEFAULT_DECODE_KEYS:
            for i, datum in enumerate(data):
                data[i] = DEFAULT_DECODE_KEYS[column](datum)
    else:
        data = table.to_pylist()
        for key in DEFAULT_DECODE_KEYS:
            if key in data[0]:
                for datum in data:
                    datum[key] = DEFAULT_DECODE_KEYS[key](datum[key])
    return data


def media_decode_helper(batches, idx=0):
    _type = arrow_decoder(batches, "media.type")[idx]
    if _type == MediaType.IMAGE:
        def image_decoder(path):
            data = arrow_decoder(batches, "media.bytes")[idx]
            return ImageFileMapper.backward(data=data)

        path = arrow_decoder(batches, "media.path")[idx]
        attributes = arrow_decoder(batches, "media.attributes")[idx]
        return Image(data=image_decoder, path=path, size=attributes["size"])
    else:
        raise NotImplementedError


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
    def backward_from_batches(batches: List[pa.lib.RecordBatch]) -> List[DatasetItem]:
        ids = arrow_decoder(batches, "id")
        subsets = arrow_decoder(batches, "subset")
        annotations_ = arrow_decoder(batches, "annotations")
        attributes_ = arrow_decoder(batches, "attributes")

        items = []
        for i, (id, subset, annotations, attributes) in enumerate(
            zip(ids, subsets, annotations_, attributes_)
        ):
            items.append(
                DatasetItem(
                    id=id,
                    subset=subset,
                    media=media_decode_helper(batches, i),
                    annotations=annotations,
                    attributes=attributes,
                )
            )
        return items
