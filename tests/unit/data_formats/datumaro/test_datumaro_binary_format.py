# pylint: disable=arguments-differ
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


from typing import Any

import numpy as np
import pytest

from datumaro.components.annotation import (
    Bbox,
    Caption,
    Cuboid3d,
    Ellipse,
    Label,
    Mask,
    Points,
    Polygon,
    PolyLine,
)
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.data_formats.datumaro_binary.crypter import Crypter
from datumaro.plugins.data_formats.datumaro_binary.exporter import DatumaroBinaryExporter
from datumaro.plugins.data_formats.datumaro_binary.importer import DatumaroBinaryImporter
from datumaro.plugins.data_formats.datumaro_binary.mapper import *

#     DatasetItemMapper,
#     DictMapper,
#     Mapper,
#     StringMapper,
#     MaskMapper,
#     BboxMapper,
# )
from datumaro.plugins.data_formats.datumaro_binary.mapper.annotation import LabelMapper

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

    def test_source_target_pair(self):
        pass

    def test_inplace_save_writes_only_updated_data_with_direct_changes(self):
        pass

    def test_inplace_save_writes_only_updated_data_with_transforms(self):
        pass


@pytest.mark.parametrize(
    "mapper,expected",
    [
        (StringMapper, "9sd#&(d!d.x]+="),
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
        (
            DatasetItemMapper,
            DatasetItem(
                id="item_0",
                subset="test",
                media=Image(path="dummy.png", size=(10, 10)),
                attributes={"x": 1, "y": 2},
            ),
        ),
        (
            DatasetItemMapper,
            DatasetItem(
                id="item_0",
                subset="test",
                media=Image(path="dummy.png", size=None),
                attributes={"x": 1, "y": 2},
            ),
        ),
        (
            LabelMapper,
            Label(
                label=1,
                id=2,
                attributes={"x": 1, "y": 2},
                group=3,
            ),
        ),
        (
            MaskMapper,
            Mask(
                label=3,
                id=5,
                z_order=2,
                image=np.ones((2, 3)),
                attributes={
                    "x": 1,
                    "y": "2",
                },
            ),
        ),
        (
            PointsMapper,
            Points(
                [1, 2, 2, 0, 1, 1],
                label=0,
                id=5,
                z_order=4,
                attributes={
                    "x": 1,
                    "y": "2",
                },
            ),
        ),
        (
            PolyLineMapper,
            PolyLine([1, 2, 3, 4, 5, 6, 7, 8], id=11, z_order=1),
        ),
        (
            PolygonMapper,
            Polygon([1, 2, 3, 4, 5, 6, 7, 8], id=12, z_order=4),
        ),
        (
            BboxMapper,
            Bbox(
                5,
                6,
                7,
                8,
                id=5,
                group=5,
                attributes={
                    "a": 1.5,
                    "b": "text",
                },
            ),
        ),
        (
            CaptionMapper,
            Caption("test"),
        ),
        (
            Cuboid3dMapper,
            Cuboid3d(
                [1.0, 2.0, 3.0],
                [2.0, 2.0, 4.0],
                [1.0, 3.0, 4.0],
                id=6,
                label=0,
                attributes={"occluded": True},
                group=6,
            ),
        ),
        (
            EllipseMapper,
            Ellipse(
                5,
                6,
                7,
                8,
                label=3,
                id=5,
                z_order=2,
                attributes={
                    "x": 1,
                    "y": "2",
                },
            ),
        ),
        (
            AnnotationListMapper,
            [
                Cuboid3d(
                    [1.0, 2.0, 3.0],
                    [2.0, 2.0, 4.0],
                    [1.0, 3.0, 4.0],
                    id=6,
                    label=0,
                    attributes={"occluded": True},
                    group=6,
                ),
                Ellipse(
                    5,
                    6,
                    7,
                    8,
                    label=3,
                    id=5,
                    z_order=2,
                    attributes={
                        "x": 1,
                        "y": "2",
                    },
                ),
            ],
        ),
    ],
)
def test_mapper(mapper: Mapper, expected: Any):
    _bytes = mapper.forward(expected)
    actual, _ = mapper.backward(_bytes)
    assert expected == actual

    prefix = bytes("asdf", "utf-8")
    offset = len(prefix)
    actual, _ = mapper.backward(prefix + _bytes, offset=offset)
    assert expected == actual
