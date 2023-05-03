# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer

from .format import SynthiaFormatType, SynthiaRandPath, SynthiaSfPath, SynthiaAlPath


class _SynthiaImporter(Importer):
    FORMAT = None
    META_FOLDERS = []

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        for folder in cls.META_FOLDERS:
            if not osp.isdir(osp.join(context.root_path, folder)):
                context.fail("Any Synthia format is not detected.")

        # with context.require_any():
        #     for prefix in cls.META_FOLDERS:
        #         with context.alternative():
        #             context.require_file(f"{prefix}/*.png")

        return FormatDetectionConfidence.MEDIUM

    @classmethod
    def find_sources(cls, path):
        return [{"url": path, "format": cls.FORMAT}]


class SynthiaRandImporter(_SynthiaImporter):
    FORMAT = SynthiaFormatType.synthia_rand.name
    META_FOLDERS = []
    for prefix in vars(SynthiaRandPath).values():
        if isinstance(prefix, str) and "datumaro" not in prefix:
            META_FOLDERS.append(prefix)


class SynthiaSfImporter(_SynthiaImporter):
    FORMAT = SynthiaFormatType.synthia_sf.name
    META_FOLDERS = []
    for prefix in vars(SynthiaSfPath).values():
        if isinstance(prefix, str) and "datumaro" not in prefix:
            META_FOLDERS.append(prefix)


class SynthiaAlImporter(_SynthiaImporter):
    FORMAT = SynthiaFormatType.synthia_al.name
    META_FOLDERS = []
    for prefix in vars(SynthiaAlPath).values():
        if isinstance(prefix, str) and "datumaro" not in prefix:
            META_FOLDERS.append(prefix)
