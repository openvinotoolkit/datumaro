# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


from datumaro.components.environment import Environment
from datumaro.components.project import Dataset
from datumaro.plugins.data_formats.datumaro_binary.exporter import DatumaroBinaryExporter
from datumaro.plugins.data_formats.datumaro_binary.importer import DatumaroBinaryImporter

from ....requirements import Requirements, mark_requirement
from .test_datumaro_format import DatumaroFormatTest as TestBase


class DatumaroBinaryFormatTest(TestBase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self, fxt_test_datumaro_format_dataset: Dataset, test_dir):
        DatumaroBinaryExporter.convert(fxt_test_datumaro_format_dataset, save_dir=test_dir)

        detected_formats = Environment().detect_dataset(test_dir)
        assert [DatumaroBinaryImporter.NAME] == detected_formats

    # Implementation has not been finished.
    # Those tests will be enabled after implementations.
    def test_can_save_and_load(self):
        pass

    def test_source_target_pair(self):
        pass

    def test_inplace_save_writes_only_updated_data_with_direct_changes(self):
        pass

    def test_inplace_save_writes_only_updated_data_with_transforms(self):
        pass
