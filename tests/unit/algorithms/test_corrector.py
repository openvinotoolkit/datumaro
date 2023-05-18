# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Dict

import pytest
import numpy as np

from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories

import datumaro.plugins.transforms as transforms
from datumaro.plugins.validators import ClassificationValidator, DetectionValidator

from tests.utils.test_utils import compare_datasets


@pytest.fixture
def fxt_dirty_cls_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="normal_train_0",
                subset="train",
                annotations=[
                    Label(id=0, label=0),
                ],
            ),
            DatasetItem(
                id="normal_train_1",
                subset="train",
                annotations=[
                    Label(id=0, label=0),
                ],
            ),
            DatasetItem(
                id="normal_train_2",
                subset="train",
                annotations=[
                    Label(id=0, label=1),
                ],
            ),
            DatasetItem(
                id="normal_train_3",
                subset="train",
                annotations=[
                    Label(id=0, label=1),
                ],
            ),
            DatasetItem(
                id="undefined_label_0",
                subset="train",
                annotations=[
                    Label(id=0, label=100),
                ],
            ),
            DatasetItem(
                id="multi_label_0",
                subset="train",
                annotations=[
                    Label(id=0, label=0),
                    Label(id=1, label=1),
                ],
            ),
            DatasetItem(
                id="few_sample_0",
                subset="train",
                annotations=[
                    Label(id=0, label=3),
                ],
            ),
            DatasetItem(
                id="missing_label_0",
                subset="train",
            ),
            DatasetItem(
                id="normal_test_0",
                subset="test",
                annotations=[
                    Label(id=0, label=0),
                ],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                "label_" + str(label) for label in range(4)
            ),
        },
    )


@pytest.fixture
def fxt_clean_cls_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="normal_train_0",
                subset="train",
                annotations=[
                    Label(id=0, label=0),
                ],
            ),
            DatasetItem(
                id="normal_train_1",
                subset="train",
                annotations=[
                    Label(id=0, label=0),
                ],
            ),
            DatasetItem(
                id="normal_train_2",
                subset="train",
                annotations=[
                    Label(id=0, label=1),
                ],
            ),
            DatasetItem(
                id="normal_train_3",
                subset="train",
                annotations=[
                    Label(id=0, label=1),
                ],
            ),
            DatasetItem(
                id="few_sample_0",
                subset="train",
                annotations=[
                    Label(id=0, label=3),
                ],
            ),
            DatasetItem(
                id="normal_test_0",
                subset="test",
                annotations=[
                    Label(id=0, label=0),
                ],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                "label_" + str(label) for label in range(4)
            ),
        },
    )


@pytest.fixture
def fxt_dirty_det_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="normal_train_0",
                subset="train",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=1),
                ],
            ),
            DatasetItem(
                id="normal_train_1",
                subset="train",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=1),
                    Bbox(50, 50, 100, 100, id=2, label=2),
                ],
            ),
            DatasetItem(
                id="undefined_label_0",
                subset="train",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=0),
                    Bbox(0, 0, 100, 100, id=2, label=100),
                ],
            ),
            DatasetItem(
                id="negative_len_0",
                subset="train",
                annotations=[
                    Bbox(0, 0, 0.5, 0.5, id=1, label=0),
                    Bbox(0, 0, 100, 100, id=2, label=1),
                ],
            ),
            DatasetItem(
                id="negative_len_1",
                subset="train",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=0),
                    Bbox(0, 0, 0, 1, id=2, label=1),
                ],
            ),
            DatasetItem(
                id="invalid_0",
                subset="train",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=0),
                    Bbox(0, 0, np.inf, 10, id=2, label=1),
                ],
            ),
            DatasetItem(
                id="missing_label_0",
                subset="train",
            ),
            DatasetItem(
                id="normal_test_0",
                subset="test",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=1),
                    Bbox(0, 0, 100, 100, id=2, label=2),
                ],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                "label_" + str(label) for label in range(4)
            ),
        },
    )


@pytest.fixture
def fxt_clean_det_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="normal_train_0",
                subset="train",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=1),
                ],
            ),
            DatasetItem(
                id="normal_train_1",
                subset="train",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=1),
                    Bbox(50, 50, 100, 100, id=2, label=2),
                ],
            ),
            DatasetItem(
                id="negative_len_0",
                subset="train",
                annotations=[
                    Bbox(0, 0, 100, 100, id=2, label=1),
                ],
            ),
            DatasetItem(
                id="negative_len_1",
                subset="train",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=0),
                ],
            ),
            DatasetItem(
                id="invalid_0",
                subset="train",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=0),
                ],
            ),
            DatasetItem(
                id="normal_test_0",
                subset="test",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=1),
                    Bbox(0, 0, 100, 100, id=2, label=2),
                ],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                "label_" + str(label) for label in range(4)
            ),
        },
    )


class ValidationCorrectionTest:
    @pytest.mark.parametrize(
        "fxt_dirty_cls_dataset, fxt_args",
        [
            ("fxt_dirty_cls_dataset", {"few_samples_thr": 1}),
        ],
        indirect=["fxt_dirty_cls_dataset"],
        ids=[
            "refine_cls_data",
        ],
    )
    def test_can_refine_cls_data_with_validation(
        self,
        fxt_dirty_cls_dataset: Dataset,
        fxt_args: Dict,
        fxt_clean_cls_dataset: Dataset,
        request: pytest.FixtureRequest,
    ):
        validator = ClassificationValidator(**fxt_args)
        reports = validator.validate(fxt_dirty_cls_dataset)

        refined = transforms.Corrector(fxt_dirty_cls_dataset, reports=reports)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, refined, fxt_clean_cls_dataset)

    @pytest.mark.parametrize(
        "fxt_dirty_det_dataset, fxt_args",
        [
            ("fxt_dirty_det_dataset", {"few_samples_thr": 1}),
        ],
        indirect=["fxt_dirty_det_dataset"],
        ids=[
            "refine_det_data",
        ],
    )
    def test_can_refine_det_data_with_validation(
        self,
        fxt_dirty_det_dataset: Dataset,
        fxt_args: Dict,
        fxt_clean_det_dataset: Dataset,
        request: pytest.FixtureRequest,
    ):
        validator = DetectionValidator(**fxt_args)
        reports = validator.validate(fxt_dirty_det_dataset)

        refined = transforms.Corrector(fxt_dirty_det_dataset, reports=reports)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, refined, fxt_clean_det_dataset)
