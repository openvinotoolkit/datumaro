from typing import List

import numpy as np
import pytest

from datumaro.components.annotation import Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.components.shift_analyzer import ShiftAnalyzer

from ..requirements import Requirements, mark_requirement


@pytest.fixture
def fxt_dataset_ideal():
    src_dataset = Dataset.from_iterable(
        [
            DatasetItem(id=0, media=Image(data=255 * np.ones((8, 8, 3))), annotations=[Label(0)]),
            DatasetItem(id=1, media=Image(data=255 * np.ones((8, 8, 3))), annotations=[Label(0)]),
            DatasetItem(id=2, media=Image(data=255 * np.ones((8, 8, 3))), annotations=[Label(1)]),
            DatasetItem(id=3, media=Image(data=255 * np.ones((8, 8, 3))), annotations=[Label(1)]),
        ],
        categories=["a", "b"],
    )

    tgt_dataset = Dataset.from_iterable(
        [
            DatasetItem(id=0, media=Image(data=255 * np.ones((8, 8, 3))), annotations=[Label(0)]),
            DatasetItem(id=1, media=Image(data=255 * np.ones((8, 8, 3))), annotations=[Label(0)]),
            DatasetItem(id=2, media=Image(data=255 * np.ones((8, 8, 3))), annotations=[Label(1)]),
            DatasetItem(id=3, media=Image(data=255 * np.ones((8, 8, 3))), annotations=[Label(1)]),
        ],
        categories=["a", "b"],
    )
    return [src_dataset, tgt_dataset]


@pytest.fixture
def fxt_dataset_different():
    src_dataset = Dataset.from_iterable(
        [
            DatasetItem(id=0, media=Image(data=255 * np.ones((8, 8, 3))), annotations=[Label(0)]),
            DatasetItem(id=1, media=Image(data=255 * np.ones((8, 8, 3))), annotations=[Label(0)]),
            DatasetItem(id=2, media=Image(data=255 * np.ones((8, 8, 3))), annotations=[Label(1)]),
            DatasetItem(id=3, media=Image(data=255 * np.ones((8, 8, 3))), annotations=[Label(1)]),
        ],
        categories=["a", "b"],
    )

    tgt_dataset = Dataset.from_iterable(
        [
            DatasetItem(id=0, media=Image(data=np.zeros((8, 8, 3))), annotations=[Label(1)]),
            DatasetItem(id=1, media=Image(data=np.zeros((8, 8, 3))), annotations=[Label(1)]),
            DatasetItem(id=2, media=Image(data=np.zeros((8, 8, 3))), annotations=[Label(1)]),
            DatasetItem(id=3, media=Image(data=np.zeros((8, 8, 3))), annotations=[Label(1)]),
        ],
        categories=["a", "b"],
    )
    return [src_dataset, tgt_dataset]


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
@pytest.mark.parametrize(
    "fxt_datasets,method,expected",
    [
        ("fxt_dataset_ideal", "fid", 0),
        ("fxt_dataset_ideal", "emd", 0),
        ("fxt_dataset_different", "fid", 0.100584),
        ("fxt_dataset_different", "emd", 0.003166),
    ],
)
def test_covariate_shift(
    fxt_datasets: List[Dataset], method: str, expected: float, request: pytest.FixtureRequest
):
    fxt_datasets = request.getfixturevalue(fxt_datasets)
    shift_analyzer = ShiftAnalyzer()
    result = shift_analyzer.compute_covariate_shift(fxt_datasets, method=method)
    assert abs(result - expected) < 1e-6


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
@pytest.mark.parametrize(
    "fxt_datasets,expected",
    [
        ("fxt_dataset_ideal", 0.75),
        ("fxt_dataset_different", 0.959047),
    ],
)
def test_label_shift(fxt_datasets: List[Dataset], expected: float, request: pytest.FixtureRequest):
    fxt_datasets = request.getfixturevalue(fxt_datasets)
    shift_analyzer = ShiftAnalyzer()
    result = shift_analyzer.compute_label_shift(fxt_datasets)
    assert abs(result - expected) < 1e-6
