# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

import pytest

from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories, Mask

from datumaro.plugins.validators import ClassificationValidator

@pytest.fixture
def fxt_errorneous_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="normal_train_0",
                subset="train",
                annotations=[
                    Label(id=0, label=0),
                    Bbox(0, 0, 100, 100, id=1, label=1),
                ],
            ),
            DatasetItem(
                id="normal_train_1",
                subset="train",
                annotations=[
                    Label(id=0, label=1),
                    Bbox(0, 0, 100, 100, id=1, label=1),
                    Bbox(50, 50, 100, 100, id=2, label=2),
                ],
            ),
            DatasetItem(
                id="undefined_label_0",
                subset="train",
                annotations=[
                    Label(id=0, label=2),
                    Bbox(0, 0, 100, 100, id=1, label=0),
                    Bbox(0, 0, 100, 100, id=2, label=100),
                ],
            ),
            DatasetItem(
                id="multi_label_0",
                subset="train",
                annotations=[
                    Label(id=0, label=0),
                    Label(id=1, label=1),
                    Bbox(0, 0, 100, 100, id=2, label=2),
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
                id="negative_len_0",
                subset="train",
                annotations=[
                    Label(id=0, label=0),
                    Bbox(0, 0, 1, 1, id=1, label=0),
                    Bbox(0, 0, 100, 100, id=2, label=1),
                ],
            ),
            DatasetItem(
                id="invalid_value_0",
                subset="train",
                annotations=[
                    Label(id=0, label=0),
                    Bbox(0, 0, 100, 100, id=1, label=0),
                    Bbox(0, 0, 0, 1, id=2, label=1),
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
    def test_can_refine_data_with_validation(self, fxt_dataset: Dataset):
        extra_args = {
            "few_samples_thr": 1,
            "imbalance_ratio_thr": 5,
            "far_from_mean_thr": 20.0,
        }

        validator = ClassificationValidator(**extra_args)
        reports = validator.validate(fxt_dataset)

        print(reports)
