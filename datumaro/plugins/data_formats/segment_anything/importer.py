# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from typing import Dict, List, Optional

from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.errors import DatasetImportError
from datumaro.util import parse_json


class SegmentAnythingImporter(Importer):
    _N_JSON_TO_TEST = 10

    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> Optional[FormatDetectionConfidence]:
        # test maximum 10 annotation files only
        ctr = 0
        for file in context.require_files_iter("*.json"):
            ctr += 1
            with context.probe_text_file(
                file, "Annotation format is not Segmentat-Anything format", is_binary_file=True
            ) as f:
                anno = parse_json(f.read())
                if (
                    set(anno.keys()) != {"annotations", "image"}
                    or (
                        set(anno["image"].keys())
                        != {
                            "image_id",
                            "width",
                            "height",
                            "file_name",
                        }
                    )
                    or (
                        anno["annotations"]
                        and not {"id", "segmentation", "bbox"}.issubset(set(anno["annotations"][0]))
                    )
                ):
                    raise DatasetImportError
            if ctr > cls._N_JSON_TO_TEST:
                break

    @classmethod
    def find_sources(cls, path) -> List[Dict]:
        if not os.path.isdir(path):
            return []
        return [{"url": path, "format": cls.NAME}]
