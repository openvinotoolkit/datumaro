# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from typing import Dict, List, Optional, Type

from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.components.merge.extractor_merger import ExtractorMerger
from datumaro.util import parse_json

from .format import DatumaroPath


class DatumaroImporter(Importer):
    PATH_CLS = DatumaroPath

    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> Optional[FormatDetectionConfidence]:
        annot_file = context.require_file(
            osp.join(cls.PATH_CLS.ANNOTATIONS_DIR, "*" + cls.PATH_CLS.ANNOTATION_EXT)
        )

        with context.probe_text_file(
            annot_file,
            'must be a JSON object with "categories" ' 'and "items" keys',
        ) as f:
            contents = parse_json(f.read())
            if not {"categories", "items"} <= contents.keys():
                raise Exception

    @classmethod
    def find_sources(cls, path) -> List[Dict]:
        return cls._find_sources_recursive(
            path,
            cls.PATH_CLS.ANNOTATION_EXT,
            cls.NAME,
            dirname=cls.PATH_CLS.ANNOTATIONS_DIR,
        )

    @property
    def can_stream(self) -> bool:
        return True

    def get_extractor_merger(self) -> Type[ExtractorMerger]:
        return ExtractorMerger
