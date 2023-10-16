# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Any, Dict

import pyarrow as pa

from datumaro.components.dataset_base import DatasetItem
from datumaro.plugins.data_formats.datumaro_binary.mapper.annotation import AnnotationListMapper
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import DictMapper, Mapper

from .media import MediaMapper


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
    def backward(idx: int, table: pa.Table, table_path: str) -> DatasetItem:
        return DatasetItem(
            id=table.column("id")[idx].as_py(),
            subset=table.column("subset")[idx].as_py(),
            media=MediaMapper.backward(
                media_struct=table.column("media")[idx],
                idx=idx,
                table=table,
                table_path=table_path,
            ),
            annotations=AnnotationListMapper.backward(table.column("annotations")[idx].as_py())[0],
            attributes=DictMapper.backward(table.column("attributes")[idx].as_py())[0],
        )
