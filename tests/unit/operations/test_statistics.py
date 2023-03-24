# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Tuple

import numpy as np
import pytest
from unittest.mock import patch
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image, PointCloud
from datumaro.components.operations import IMAGE_STATS_SCHEMA, compute_image_statistics

from tests.requirements import Requirements, mark_requirement


@pytest.fixture
def fxt_image_dataset_expected_mean_std():
    expected_mean = [100, 50, 150]
    expected_std = [20, 50, 10]

    return expected_mean, expected_std


@pytest.fixture
def fxt_image_dataset(fxt_image_dataset_expected_mean_std: Tuple[List[int], List[int]]):
    np.random.seed(3003)

    expected_mean, expected_std = fxt_image_dataset_expected_mean_std

    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id=i,
                media=Image(data=np.random.normal(expected_mean, expected_std, size=(h, w, 3))),
            )
            for i, (w, h) in enumerate([(3000, 100), (800, 600), (400, 200), (700, 300)])
        ]
    )
    dataset.put(dataset.get("1"), id="5", subset="train")
    return dataset


@pytest.fixture
def fxt_point_cloud_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id=i,
                media=PointCloud(path="dummy.pcd"),
            )
            for i in range(5)
        ],
        media_type=PointCloud,
    )
    return dataset


class ImageStatisticsTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_image_stats(
        self,
        fxt_image_dataset: Dataset,
        fxt_image_dataset_expected_mean_std: Tuple[List[int], List[int]],
    ):
        expected_mean, expected_std = fxt_image_dataset_expected_mean_std
        actual = compute_image_statistics(fxt_image_dataset)

        assert actual["dataset"] == {
            "images count": 5,
            "unique images count": 4,
            "repeated images count": 1,
            "repeated images": [[("1", "default"), ("5", "train")]],
        }
        assert actual["subsets"]["default"]["images count"] == 4
        assert actual["subsets"]["train"]["images count"] == 1

        actual_mean = actual["subsets"]["default"]["image mean"][::-1]
        actual_std = actual["subsets"]["default"]["image std"][::-1]

        for em, am in zip(expected_mean, actual_mean):
            assert am == pytest.approx(em, 1e-2)
        for estd, astd in zip(expected_std, actual_std):
            assert astd == pytest.approx(estd, 1e-2)

    @mark_requirement(Requirements.DATUM_BUG_873)
    def test_invalid_media_type(
        self,
        fxt_point_cloud_dataset: Dataset,
    ):
        # PointCloud media_type at dataset level
        with pytest.raises(DatumaroError, match="only Image media_type is allowed"):
            actual = compute_image_statistics(fxt_point_cloud_dataset)

        # Exceptional case of #873, dataset has Image media_type but DatasetItem has PointCloud.
        with patch.object(Dataset, "media_type", return_value=Image):
            with pytest.warns(UserWarning, match="only Image media_type is allowed"):
                actual = compute_image_statistics(fxt_point_cloud_dataset)
            assert actual["dataset"] == IMAGE_STATS_SCHEMA["dataset"]
