# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from typing import List, Optional

from datumaro.components.errors import DatasetImportError
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image, PointCloud
from datumaro.util import cast
from datumaro.util.image import find_images
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file

from .format import Kitti3DPath, OcclusionStates, TruncationStates


class Kitti3dImporter(Importer):
    _ANNO_EXT = ".txt"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        with context.require_any():
            with context.alternative():
                cls._check_ann_file(context.require_file(f"{Kitti3DPath.LABEL_DIR}/*.txt"), context)

        return FormatDetectionConfidence.MEDIUM

    @classmethod
    def _check_ann_file(cls, fpath: str, context: FormatDetectionContext) -> bool:
        with context.probe_text_file(
            fpath, "Requirements for the annotation file of Kitti 3D format"
        ) as fp:
            for line in fp:
                fields = line.rstrip("\n").split(" ")
                if len(fields) == 15 or len(fields) == 16:
                    return True
                raise DatasetImportError(
                    f"Kitti 3D format txt file should have 15 or 16 fields for "
                    f"each line, but the read line has {len(fields)} fields: "
                    f"fields={fields}."
                )
            raise DatasetImportError("Empty file is not allowed.")

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]
