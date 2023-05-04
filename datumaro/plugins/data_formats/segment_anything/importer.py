# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json
import os
from typing import Dict, List, Optional

from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.errors import DatasetImportError


class SegmentAnythingImporter(Importer):
    _N_TEST_JSON = 10

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
                file, "Annotation format is not Segmentat-Anything format"
            ) as f:
                anno = json.load(f)
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
            if ctr > cls._N_TEST_JSON:
                break

    @classmethod
    def find_sources(cls, path) -> List[Dict]:
        if not os.path.isdir(path):
            return []
        return [{"url": path, "format": cls.NAME}]
