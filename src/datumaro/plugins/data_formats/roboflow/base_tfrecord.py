# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import re
from typing import Optional

from datumaro.components.importer import ImportContext
from datumaro.plugins.data_formats.tf_detection_api.base import TfDetectionApiBase
from datumaro.plugins.data_formats.tf_detection_api.format import TfrecordImporterType


class RoboflowTfrecordBase(TfDetectionApiBase):
    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(
            path=path,
            subset=subset,
            tfrecord_importer_type=TfrecordImporterType.roboflow,
            ctx=ctx,
        )

    @staticmethod
    def _parse_labelmap(text):
        entry_pattern = r'name:\s*"([^"]+)"\s*,\s*id:\s*(\d+)'
        entry_pattern = re.compile(entry_pattern)

        matches = re.findall(entry_pattern, text)

        labelmap = {name: int(id) for name, id in matches}

        return labelmap
