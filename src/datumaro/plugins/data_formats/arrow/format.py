# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import struct
import pyarrow as pa
from datumaro.components.dataset_base import IDataset
from datumaro.errors import DatasetImportError
from datumaro.plugins.data_formats.datumaro.exporter import JsonWriter
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import DictMapper


class DatumaroArrow:
    SIGNATURE = "signature:datumaro_arrow"
    VERSION = "2.0"

    MP_TIMEOUT = 300.0  # 5 minutes

    ID_FIELD = "id"
    SUBSET_FIELD = "subset"
    MEDIA_FIELD = "media"

    SCHEMA = pa.schema(
        [
            pa.field(ID_FIELD, pa.string()),
            pa.field(SUBSET_FIELD, pa.string()),
            # pa.field("media_type", pa.uint32()),
            # pa.field("media_path", pa.string()),
            # pa.field("media_bytes", pa.binary()),
            # pa.field("media_attributes", pa.binary()),
            pa.field("media_type", pa.uint32()),
            pa.field("media_has_bytes", pa.bool_()),
            pa.field("media_path", pa.string()),
            pa.field("media_bytes", pa.binary()),
            pa.field("media_attributes", pa.binary()),
            pa.field("annotations", pa.binary()),
            pa.field("attributes", pa.binary()),
        ]
    )

    @classmethod
    def check_signature(cls, signature: str):
        if signature != cls.SIGNATURE:
            raise DatasetImportError(
                f"Input signature={signature} is not aligned with the ground truth signature={cls.SIGNATURE}"
            )

    @classmethod
    def check_schema(cls, schema: pa.lib.Schema):
        if not cls.SCHEMA.equals(schema):
            raise DatasetImportError(
                f"input schema is not aligned with the ground truth schema.\n"
                f"input schema: \n{schema.to_string()}\n"
                f"ground truth schema: \n{cls.SCHEMA.to_string()}\n"
            )

    @classmethod
    def check_version(cls, version: str):
        if version != cls.VERSION:
            raise DatasetImportError(
                f"Input version={version} is not aligned with the current data format version={cls.VERSION}"
            )

    @classmethod
    def create_schema_with_metadata(cls, extractor: IDataset):
        media_type = extractor.media_type()._type
        categories = JsonWriter.write_categories(extractor.categories())

        return cls.SCHEMA.with_metadata(
            {
                "signature": cls.SIGNATURE,
                "version": cls.VERSION,
                "infos": DictMapper.forward(extractor.infos()),
                "categories": DictMapper.forward(categories),
                "media_type": struct.pack("<I", int(media_type)),
            }
        )
