# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=no-self-use

from io import TextIOWrapper

from datumaro.plugins.data_formats.datumaro.exporter import DatumaroExporter
from datumaro.plugins.data_formats.datumaro.exporter import _SubsetWriter as __SubsetWriter

from .format import DatumaroBinaryPath


class _SubsetWriter(__SubsetWriter):
    """"""

    def _sign(self, fp: TextIOWrapper):
        fp.write(DatumaroBinaryPath.SIGNATURE.encode("utf-8"))

    def write(self):
        with open(self.ann_file, "wb") as fp:
            self._sign(fp)


class DatumaroBinaryExporter(DatumaroExporter):
    DEFAULT_IMAGE_EXT = DatumaroBinaryPath.IMAGE_EXT
    WRITER_CLS = _SubsetWriter
    PATH_CLS = DatumaroBinaryPath
