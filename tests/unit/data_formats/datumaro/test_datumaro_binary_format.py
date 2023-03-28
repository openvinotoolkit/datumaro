# pylint: disable=arguments-differ
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import sys
from typing import Any

import pytest

from datumaro.components.annotation import Annotation
from datumaro.components.crypter import Crypter
from datumaro.plugins.data_formats.datumaro_binary import (
    DatumaroBinaryExporter,
    DatumaroBinaryImporter,
)
from datumaro.plugins.data_formats.datumaro_binary.format import DatumaroBinaryPath

# pylint: disable=undefined-variable
# We have to disable this because IntListMapper and FloatListMapper are imported by the following line,
# but Pylint raises the following errors.
# tests/unit/data_formats/datumaro/test_datumaro_binary_format.py:107:16: E0602: Undefined variable 'IntListMapper' (undefined-variable)
# tests/unit/data_formats/datumaro/test_datumaro_binary_format.py:115:16: E0602: Undefined variable 'FloatListMapper' (undefined-variable)
from datumaro.plugins.data_formats.datumaro_binary.mapper import *
from datumaro.plugins.data_formats.datumaro_binary.mapper.annotation import AnnotationMapper

from .test_datumaro_format import DatumaroFormatTest as TestBase

from tests.utils.test_utils import compare_datasets_strict

ENCRYPTION_KEY = Crypter.gen_key()


class DatumaroBinaryFormatTest(TestBase):
    exporter = DatumaroBinaryExporter
    importer = DatumaroBinaryImporter
    format = DatumaroBinaryImporter.NAME
    ann_ext = DatumaroBinaryPath.ANNOTATION_EXT

    # Implementation has not been finished.
    # Those tests will be enabled after implementations.
    @pytest.mark.parametrize(
        ["fxt_dataset", "compare", "require_media", "fxt_import_kwargs", "fxt_export_kwargs"],
        [
            pytest.param(
                "fxt_test_datumaro_format_dataset",
                compare_datasets_strict,
                True,
                {},
                {},
                id="test_no_encryption",
            ),
            pytest.param(
                "fxt_test_datumaro_format_dataset",
                compare_datasets_strict,
                True,
                {"encryption_key": ENCRYPTION_KEY},
                {"encryption_key": ENCRYPTION_KEY},
                id="test_with_encryption",
            ),
            pytest.param(
                "fxt_test_datumaro_format_dataset",
                compare_datasets_strict,
                True,
                {"encryption_key": ENCRYPTION_KEY},
                {"encryption_key": ENCRYPTION_KEY, "no_media_encryption": True},
                id="test_no_media_encryption",
            ),
            pytest.param(
                "fxt_test_datumaro_format_dataset",
                compare_datasets_strict,
                True,
                {"encryption_key": ENCRYPTION_KEY},
                {"encryption_key": ENCRYPTION_KEY, "max_blob_size": 1},  # 1 byte
                id="test_multi_blobs",
            ),
            pytest.param(
                "fxt_test_datumaro_format_dataset",
                compare_datasets_strict,
                True,
                {"encryption_key": ENCRYPTION_KEY, "num_workers": 2},
                {"encryption_key": ENCRYPTION_KEY, "num_workers": 2},
                id="test_multi_processing",
            ),
        ],
    )
    def test_dm_binary_own_features(
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


class MapperTest:
    @staticmethod
    def _test(mapper: Mapper, expected: Any):
        _bytes = mapper.forward(expected)
        actual, _ = mapper.backward(_bytes)
        assert expected == actual

        prefix = bytes("asdf4312", "utf-8")
        suffix = bytes("qwer5332", "utf-8")
        offset = len(prefix)
        _bytes = prefix + _bytes + suffix
        actual, offset = mapper.backward(_bytes, offset=offset)

        assert expected == actual
        assert offset == len(_bytes) - len(suffix)

    @staticmethod
    def _get_ann_mapper(ann: Annotation) -> AnnotationMapper:
        name = ann.__class__.__name__
        return getattr(sys.modules[__name__], name + "Mapper")

    @pytest.mark.parametrize(
        "mapper,expected",
        [
            (
                StringMapper,
                "9sd#&(d!d.x]+=",
            ),
            (
                IntListMapper,
                (
                    0,
                    1,
                    2,
                ),
            ),
            (
                FloatListMapper,
                (
                    0.0,
                    1.0,
                    2.0,
                ),
            ),
            (
                DictMapper,
                {
                    "string": "test",
                    "int": 0,
                    "float": 0.0,
                    "string_list": ["test0", "test1", "test2"],
                    "int_list": [0, 1, 2],
                    "float_list": [0.0, 0.1, 0.2],
                },
            ),
        ],
    )
    def test_common_mapper(self, mapper: Mapper, expected: Any):
        self._test(mapper, expected)

    def test_annotations_mapper(self, fxt_test_datumaro_format_dataset):
        """Test all annotations in fxt_test_datumaro_format_dataset"""
        mapper = DatasetItemMapper
        for item in fxt_test_datumaro_format_dataset:
            for ann in item.annotations:
                mapper = self._get_ann_mapper(ann)
                self._test(mapper, ann)


class EncryptActionTest:
    @pytest.mark.parametrize(
        "args,flag", [(["--encryption"], True), ([], False)], ids=["on", "off"]
    )
    def test_action(self, args, flag):
        parser = DatumaroBinaryExporter.build_cmdline_parser()
        args = parser.parse_args(args)

        assert hasattr(args, "encryption_key") == flag
