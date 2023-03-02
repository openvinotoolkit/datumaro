# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from typing import Optional

from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.plugins.data_formats.datumaro.importer import DatumaroImporter

from .format import DatumaroBinaryPath


class DatumaroBinaryImporter(DatumaroImporter):
    PATH_CLS = DatumaroBinaryPath

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--encryption-key",
            type=str,
            default=None,
            help="Encryption key",
        )
        return parser

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
                is_binary_file=True,
            ) as f:
                signature = f.read(DatumaroBinaryPath.SIGNATURE_LEN)
                signature = signature.decode()
                DatumaroBinaryPath.check_signature(signature)
