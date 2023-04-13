# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import re
from typing import Dict, List, Optional

import pyarrow as pa

from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer

from .format import DatumaroArrow


class ArrowImporter(Importer):
    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> Optional[FormatDetectionConfidence]:
        def verify_datumaro_arrow_format(file):
            with pa.ipc.open_stream(file) as reader:
                schema = reader.schema
            DatumaroArrow.check_signature(schema.metadata.get(b"signature", b"").decode())
            DatumaroArrow.check_schema(schema)

        if context.root_path.endswith(".arrow"):
            verify_datumaro_arrow_format(context.root_path)
        else:
            for arrow_file in context.require_files("*.arrow"):
                with context.probe_text_file(
                    arrow_file,
                    f"{arrow_file} is not Datumaro arrow format.",
                    is_binary_file=True,
                ) as f:
                    verify_datumaro_arrow_format(f)

    @classmethod
    def find_sources(cls, path) -> List[Dict]:
        sources = cls._find_sources_recursive(
            path,
            ".arrow",
            cls.NAME,
        )

        # handle sharded arrow files
        _sources = []
        for source in sources:
            match = re.match(r"(.*?)(-[0-9]+-of-[0-9]+)?\.arrow", source["url"])
            found = False
            prefix = match.group(1)
            for _source in _sources:
                if _source["url"].startswith(prefix):
                    _source["options"]["additional_paths"].append(source["url"])
                    found = True
                    break
            if not found:
                source["options"] = {
                    "additional_paths": [],
                    "subset": os.path.basename(prefix),
                }
                _sources.append(source)
        return _sources
