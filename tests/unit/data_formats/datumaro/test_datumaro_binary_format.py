# pylint: disable=arguments-differ
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import pytest

from datumaro.components.environment import Environment
from datumaro.components.project import Dataset
from datumaro.plugins.data_formats.datumaro_binary.crypter import Crypter
from datumaro.plugins.data_formats.datumaro_binary.exporter import DatumaroBinaryExporter
from datumaro.plugins.data_formats.datumaro_binary.importer import DatumaroBinaryImporter
from datumaro.util.test_utils import compare_datasets_strict

from ....requirements import Requirements, mark_requirement
from .test_datumaro_format import DatumaroFormatTest as TestBase


def tmp_compare(test, expected, actual, *args, **kwargs):
    assert expected.infos() == actual.infos()
    assert expected.categories() == actual.categories()


ENCRYPTION_KEY = Crypter.gen_key()


class DatumaroBinaryFormatTest(TestBase):
    exporter = DatumaroBinaryExporter
    importer = DatumaroBinaryImporter

    # Implementation has not been finished.
    # Those tests will be enabled after implementations.
    @pytest.mark.parametrize(
        ["fxt_dataset", "compare", "require_media", "fxt_import_kwargs", "fxt_export_kwargs"],
        [
            pytest.param(
                "fxt_test_datumaro_format_dataset",
                tmp_compare,
                True,
                {},
                {},
                id="test_no_encryption",
            ),
            pytest.param(
                "fxt_test_datumaro_format_dataset",
                tmp_compare,
                True,
                {"encryption_key": ENCRYPTION_KEY},
                {"encryption_key": ENCRYPTION_KEY},
                id="test_with_encryption",
            ),
        ],
    )
    def test_can_save_and_load(
        self,
        fxt_dataset,
        compare,
        require_media,
        test_dir,
        fxt_import_kwargs,
        fxt_export_kwargs,
        helper_tc,
        request,
    ):
        return super().test_can_save_and_load(
            fxt_dataset,
            compare,
            require_media,
            test_dir,
            fxt_import_kwargs,
            fxt_export_kwargs,
            helper_tc,
            request,
        )

    def test_source_target_pair(self):
        pass

    def test_inplace_save_writes_only_updated_data_with_direct_changes(self):
        pass

    def test_inplace_save_writes_only_updated_data_with_transforms(self):
        pass
