# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from typing import List

from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer

from .format import SynthiaAlPath, SynthiaFormatType, SynthiaRandPath, SynthiaSfPath


class _SynthiaImporter(Importer):
    FORMAT = None
    META_FOLDERS = []

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        for folder in cls.META_FOLDERS:
            if not osp.isdir(osp.join(context.root_path, folder)):
                context.fail("Any Synthia format is not detected.")

        return FormatDetectionConfidence.MEDIUM

    @classmethod
    def find_sources(cls, path):
        return [{"url": path, "format": cls.FORMAT}]

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [".png"]


class SynthiaRandImporter(_SynthiaImporter):
    FORMAT = SynthiaFormatType.synthia_rand.name
    META_FOLDERS = SynthiaRandPath.meta_folders()


class SynthiaSfImporter(_SynthiaImporter):
    FORMAT = SynthiaFormatType.synthia_sf.name
    META_FOLDERS = SynthiaSfPath.meta_folders()


class SynthiaAlImporter(_SynthiaImporter):
    FORMAT = SynthiaFormatType.synthia_al.name
    META_FOLDERS = SynthiaAlPath.meta_folders()
