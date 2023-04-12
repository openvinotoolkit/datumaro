# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import pyarrow as pa

from datumaro.errors import DatasetImportError


class DatumaroArrow:
    SIGNATURE = "signature:datumaro_arrow"
    VERSION = "1.0"

    MP_TIMEOUT = 300.0  # 5 minutes

    SCHEMA = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("subset", pa.string()),
            pa.field(
                "media",
                pa.struct(
                    [
                        pa.field("type", pa.uint32()),
                        pa.field("path", pa.string()),
                        pa.field("bytes", pa.binary()),
                        pa.field("attributes", pa.binary()),
                    ]
                ),
            ),
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
