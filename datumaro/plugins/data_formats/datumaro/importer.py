# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from typing import Dict, List, Optional

from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer
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
            cls.get_extractor_name(),
            dirname=cls.PATH_CLS.ANNOTATIONS_DIR,
        )
