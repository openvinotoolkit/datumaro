# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from typing import Any, Dict, Optional

import pytest

from datumaro.components.annotation import Bbox, RleMask
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.exporter import Exporter
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.plugins.data_formats.segment_anything import (
    SegmentAnythingExporter,
    SegmentAnythingImporter,
)

from ...requirements import Requirements, mark_requirement
from .base import TestDataFormatBase

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets

DATASET_DIR = get_test_asset_path("segment_anything_dataset")


@pytest.fixture
def fxt_dataset():
    expected_dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="a",
                media=Image.from_file(os.path.join(DATASET_DIR, "a.jpg")),
                annotations=[
                    Bbox(
                        3.0,
                        1.0,
                        1.0,
                        4.0,
                        label=None,
                        id=1,
                        attributes={
                            "predicted_iou": 0.2633798018925796,
                            "stability_score": 0.47182999738062237,
                            "point_coords": [[3, 4]],
                            "crop_box": [3.0, 1.0, 2.0, 5.0],
                        },
                        group=1,
                    ),
                    RleMask(
                        rle={"size": [5, 10], "counts": "`04n0"},
                        label=None,
                        id=1,
                        attributes={
                            "predicted_iou": 0.2633798018925796,
                            "stability_score": 0.47182999738062237,
                            "point_coords": [[3, 4]],
                            "crop_box": [3.0, 1.0, 2.0, 5.0],
                        },
                        group=1,
                    ),
                ],
                attributes={"id": 1},
            ),
            DatasetItem(
                id="b",
                media=Image.from_file(os.path.join(DATASET_DIR, "b.jpg")),
                annotations=[
                    Bbox(
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        label=None,
                        id=1,
                        attributes={
                            "predicted_iou": 0.8607975925446022,
                            "stability_score": 0.36768932422921985,
                            "point_coords": [[3, 8]],
                            "crop_box": [0.0, 0.0, 1.0, 1.0],
                        },
                        group=1,
                    ),
                    RleMask(
                        rle={"size": [5, 10], "counts": "b1"},
                        label=None,
                        id=1,
                        attributes={
                            "predicted_iou": 0.8607975925446022,
                            "stability_score": 0.36768932422921985,
                            "point_coords": [[3, 8]],
                            "crop_box": [0.0, 0.0, 1.0, 1.0],
                        },
                        group=1,
                    ),
                ],
                attributes={"id": 2},
            ),
        ]
    )

    return expected_dataset


class SegmentAnythingTest(TestDataFormatBase):
    IMPORTER = SegmentAnythingImporter
    EXPORTER = SegmentAnythingExporter
    USE_TEST_CAN_EXPORT_AND_IMPORT = False

    @pytest.mark.parametrize(
        "fxt_dataset_dir",
        [DATASET_DIR],
    )
    def test_can_detect(self, fxt_dataset_dir: str):
        assert DEFAULT_ENVIRONMENT.detect_dataset(fxt_dataset_dir) == ["segment_anything"]

    @pytest.mark.parametrize(
        [
            "fxt_dataset_dir",
            "fxt_expected_dataset",
            "fxt_import_kwargs",
        ],
        [
            (DATASET_DIR, "fxt_dataset", {}),
        ],
        indirect=["fxt_expected_dataset"],
    )
    def test_can_import(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        fxt_import_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
    ):
        return super().test_can_import(
            fxt_dataset_dir,
            fxt_expected_dataset,
            fxt_import_kwargs,
            request,
        )

    @pytest.mark.parametrize(
        [
            "fxt_expected_dataset",
        ],
        [
            ("fxt_dataset",),
        ],
        indirect=["fxt_expected_dataset"],
    )
    def test_can_export_and_import(
        self,
        fxt_expected_dataset: Dataset,
        test_dir: str,
        fxt_import_kwargs: Dict[str, Any],
        fxt_export_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
    ):
        return super().test_can_export_and_import(
            fxt_expected_dataset,
            test_dir,
            fxt_import_kwargs,
            fxt_export_kwargs,
            request,
        )
