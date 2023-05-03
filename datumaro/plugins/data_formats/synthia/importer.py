# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer

from .format import SynthiaFormatType, SynthiaRandPath, SynthiaSfPath, SynthiaAlPath


class _SynthiaImporter(Importer):
    FORMAT = None
    META_FILES = []

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        with context.require_any():
            for prefix in cls.META_FILES:
                with context.alternative():
                    context.require_file(f"{prefix}/*.png")

        return FormatDetectionConfidence.MEDIUM

    @classmethod
    def find_sources(cls, path):
        return [{"url": path, "format": cls.FORMAT}]


class SynthiaRandImporter(_SynthiaImporter):
    FORMAT = SynthiaFormatType.synthia_rand.name
    META_FILES = []
    for prefix in vars(SynthiaRandPath).values():
        if isinstance(prefix, str):
            META_FILES.append(prefix)


class SynthiaSfImporter(_SynthiaImporter):
    FORMAT = SynthiaFormatType.synthia_sf.name
    META_FILES = []
    for prefix in vars(SynthiaSfPath).values():
        if isinstance(prefix, str):
            META_FILES.append(prefix)


class SynthiaAlImporter(_SynthiaImporter):
    FORMAT = SynthiaFormatType.synthia_al.name
    META_FILES = []
    for prefix in vars(SynthiaAlPath).values():
        if isinstance(prefix, str):
            META_FILES.append(prefix)
