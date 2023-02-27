# pylint: disable=arguments-differ
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import sys
from typing import Any

import pytest

from datumaro.components.annotation import Annotation
from datumaro.plugins.data_formats.datumaro_binary import *
from datumaro.plugins.data_formats.datumaro_binary.crypter import Crypter
from datumaro.plugins.data_formats.datumaro_binary.mapper import *
from datumaro.plugins.data_formats.datumaro_binary.mapper.annotation import AnnotationMapper

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
    def test_develop(
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

    def test_inplace_save_writes_only_updated_data_with_direct_changes(self):
        pass

    def test_inplace_save_writes_only_updated_data_with_transforms(self):
        pass


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
