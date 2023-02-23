# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Dict, List, Optional
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.util import parse_json


class DatumaroBinaryImporter(Importer):
    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> Optional[FormatDetectionConfidence]:
        annot_file = context.require_file("annotations/*.json")

        with context.probe_text_file(
            annot_file,
            'must be a JSON object with "categories" ' 'and "items" keys',
        ) as f:
            contents = parse_json(f.read())
            if not {"categories", "items"} <= contents.keys():
                raise Exception

    @classmethod
    def find_sources(cls, path) -> List[Dict]:
        return cls._find_sources_recursive(path, ".json", "datumaro", dirname="annotations")
