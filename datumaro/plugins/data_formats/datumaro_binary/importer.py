# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Dict, List, Optional
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer
from .format import DatumaroBinaryPath
import os.path as osp


class DatumaroBinaryImporter(Importer):
    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> Optional[FormatDetectionConfidence]:
        annot_files = context.require_files(
            osp.join(DatumaroBinaryPath.ANNOTATIONS_DIR, "*" + DatumaroBinaryPath.ANNOTATION_EXT)
        )

        for annot_file in annot_files:
            with context.probe_text_file(
                annot_file,
                f"{annot_file} has no Datumaro binary format signature",
            ) as f:
                signature = f.read(len(DatumaroBinaryPath.SIGNATURE))
                if signature != DatumaroBinaryPath.SIGNATURE:
                    raise Exception()

    @classmethod
    def find_sources(cls, path: str) -> List[Dict]:
        return cls._find_sources_recursive(
            path,
            DatumaroBinaryPath.ANNOTATION_EXT,
            cls.extractor_name,
            dirname=DatumaroBinaryPath.ANNOTATIONS_DIR,
        )
