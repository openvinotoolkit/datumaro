# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from copy import deepcopy

import numpy as np
import pytest

import datumaro.plugins.transforms as transforms
from datumaro.components.annotation import AnnotationType, Bbox, Caption, Label, LabelCategories
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.media import Image, Table, TableRow
from datumaro.components.validator import Validator
from datumaro.plugins.validators import (
    ClassificationValidator,
    DetectionValidator,
    TabularValidator,
)

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import _compare_annotations, compare_datasets


@pytest.fixture
def fxt_dummy_dataset():
    yield Dataset.from_iterable([])


@pytest.fixture
def fxt_refined_dataset(request: pytest.FixtureRequest):
    fxt_name = getattr(request, "param", "fxt_dummy_dataset")
    yield request.getfixturevalue(fxt_name)


@pytest.fixture
def fxt_original_dataset(request: pytest.FixtureRequest):
    fxt_name = getattr(request, "param", "fxt_dummy_dataset")
    yield request.getfixturevalue(fxt_name)


@pytest.fixture
def fxt_original_missing_cat_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="normal_train_0",
                subset="train",
                annotations=[
                    Label(id=0, label=0, attributes={"x": 0}),
                ],
            ),
            DatasetItem(
                id="normal_test_0",
                subset="test",
                annotations=[
                    Label(id=0, label=1, attributes={"y": 1}),
                ],
            ),
        ],
    )


@pytest.fixture
def fxt_refined_missing_cat_dataset(fxt_original_missing_cat_dataset):
    refined = deepcopy(fxt_original_missing_cat_dataset)
    refined.categories()[AnnotationType.label] = LabelCategories.from_iterable(
        [
            ("0", "", ["x"]),
            ("1", "", ["y"]),
        ]
    )
    return refined


@pytest.fixture
def fxt_original_missing_label_dataset():
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
                id="missing_label_0",
                subset="train",
            ),
            DatasetItem(
                id="normal_test_0",
                subset="test",
                annotations=[
                    Label(id=0, label=1),
                ],
            ),
        ],
        categories=["a", "b"],
    )


@pytest.fixture
def fxt_refined_missing_label_dataset(fxt_original_missing_label_dataset: Dataset):
    refined = deepcopy(fxt_original_missing_label_dataset)
    refined.remove("missing_label_0", "train")
    return refined


@pytest.fixture
def fxt_original_missing_attr_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="normal_train_0",
                subset="train",
                annotations=[
                    Label(id=0, label=0, attributes={"x": 0, "y": 1}),
                ],
            ),
            DatasetItem(
                id="missing_attr_0",
                subset="train",
                annotations=[
                    Label(id=0, label=0, attributes={"x": 0}),
                ],
            ),
            DatasetItem(
                id="missing_attr_1",
                subset="train",
                annotations=[
                    Label(id=0, label=1, attributes={"y": 0}),
                ],
            ),
            DatasetItem(
                id="normal_test_0",
                subset="test",
                annotations=[
                    Label(id=0, label=1, attributes={"x": 0, "y": 1}),
                ],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                [
                    ("a", "", ["x", "y"]),
                    ("b", "", ["x", "y"]),
                ]
            ),
        },
    )


@pytest.fixture
def fxt_refined_missing_attr_dataset(fxt_original_missing_attr_dataset: Dataset):
    refined = deepcopy(fxt_original_missing_attr_dataset)
    refined.put(
        DatasetItem(
            id="missing_attr_0",
            subset="train",
            media=Image.from_numpy(data=np.ones((3, 2, 3))),
            annotations=[
                Label(id=0, label=0, attributes={"x": 0, "y": ""}),
            ],
        )
    )
    refined.put(
        DatasetItem(
            id="missing_attr_1",
            subset="train",
            media=Image.from_numpy(data=np.ones((3, 2, 3))),
            annotations=[
                Label(id=0, label=1, attributes={"x": "", "y": 0}),
            ],
        )
    )
    return refined


@pytest.fixture
def fxt_original_multi_label_dataset():
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
                id="multi_label_0",
                subset="train",
                annotations=[
                    Label(id=0, label=0),
                    Label(id=1, label=1),
                ],
            ),
            DatasetItem(
                id="normal_test_0",
                subset="test",
                annotations=[
                    Label(id=0, label=1),
                ],
            ),
        ],
        categories=["a", "b"],
    )


@pytest.fixture
def fxt_refined_multi_label_dataset(fxt_original_multi_label_dataset: Dataset):
    refined = deepcopy(fxt_original_multi_label_dataset)
    refined.remove("multi_label_0", "train")
    return refined


@pytest.fixture
def fxt_original_undefined_label_dataset():
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
                id="undefined_label_0",
                subset="train",
                annotations=[
                    Label(id=0, label=100),
                ],
            ),
            DatasetItem(
                id="normal_test_0",
                subset="test",
                annotations=[
                    Label(id=0, label=1),
                ],
            ),
        ],
        categories=["a", "b"],
    )


@pytest.fixture
def fxt_refined_undefined_label_dataset(fxt_original_undefined_label_dataset):
    refined = deepcopy(fxt_original_undefined_label_dataset)
    refined.categories()[AnnotationType.label] = LabelCategories.from_iterable(["a", "b", "100"])
    return refined


@pytest.fixture
def fxt_original_undefined_attr_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="normal_train_0",
                subset="train",
                annotations=[
                    Label(id=0, label=0, attributes={"x": 1}),
                ],
            ),
            DatasetItem(
                id="undefined_attribute_0",
                subset="train",
                annotations=[
                    Label(id=0, label=1, attributes={"z": 0}),
                ],
            ),
            DatasetItem(
                id="normal_test_0",
                subset="test",
                annotations=[
                    Label(id=0, label=1, attributes={"x": 0, "y": 2}),
                ],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                [
                    ("a", "", ["x"]),
                    ("b", "", ["x", "y"]),
                ]
            ),
        },
    )


@pytest.fixture
def fxt_refined_undefined_attr_dataset(fxt_original_undefined_attr_dataset):
    refined = deepcopy(fxt_original_undefined_attr_dataset)
    refined.put(
        DatasetItem(
            id="undefined_attribute_0",
            subset="train",
            media=Image.from_numpy(data=np.ones((3, 2, 3))),
            annotations=[
                Label(id=0, label=1, attributes={"x": "", "y": "", "z": 0}),
            ],
        )
    )
    refined.put(
        DatasetItem(
            id="normal_test_0",
            subset="test",
            media=Image.from_numpy(data=np.ones((3, 2, 3))),
            annotations=[
                Label(id=0, label=1, attributes={"x": 0, "y": 2, "z": ""}),
            ],
        )
    )
    refined.categories()[AnnotationType.label] = LabelCategories.from_iterable(
        [
            ("a", "", ["x"]),
            ("b", "", ["x", "y", "z"]),
        ]
    )
    return refined


@pytest.fixture
def fxt_original_neg_len_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="normal_train_0",
                subset="train",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=0),
                    Bbox(0, 0, 100, 100, id=2, label=1),
                ],
            ),
            DatasetItem(
                id="negative_len_0",
                subset="train",
                annotations=[
                    Bbox(0, 0, 0, 0, id=1, label=0),
                ],
            ),
            DatasetItem(
                id="normal_test_0",
                subset="test",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=0),
                    Bbox(0, 0, 100, 100, id=2, label=1),
                ],
            ),
        ],
        categories=["a", "b"],
    )


@pytest.fixture
def fxt_refined_neg_len_dataset(fxt_original_neg_len_dataset):
    refined = deepcopy(fxt_original_neg_len_dataset)
    refined.put(
        DatasetItem(
            id="negative_len_0",
            subset="train",
            media=Image.from_numpy(data=np.ones((3, 2, 3))),
        )
    )
    return refined


@pytest.fixture
def fxt_original_invalid_val_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="normal_train_0",
                subset="train",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=0),
                    Bbox(0, 0, 100, 100, id=2, label=1),
                ],
            ),
            DatasetItem(
                id="invalid_val_0",
                subset="train",
                annotations=[
                    Bbox(0, 0, np.inf, 100, id=1, label=0),
                    Bbox(0, 0, 100, np.nan, id=2, label=1),
                ],
            ),
            DatasetItem(
                id="normal_test_0",
                subset="test",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=0),
                    Bbox(0, 0, 100, 100, id=2, label=1),
                ],
            ),
        ],
        categories=["a", "b"],
    )


@pytest.fixture
def fxt_refined_invalid_val_dataset(fxt_original_invalid_val_dataset):
    refined = deepcopy(fxt_original_invalid_val_dataset)
    refined.put(
        DatasetItem(
            id="invalid_val_0",
            subset="train",
            media=Image.from_numpy(data=np.ones((3, 2, 3))),
        )
    )
    return refined


@pytest.fixture
def fxt_original_far_from_mean_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="normal_train_0",
                subset="train",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=0),
                    Bbox(0, 0, 100, 100, id=2, label=1),
                ]
                * 100,
            ),
            DatasetItem(
                id="far_from_mean_0",
                subset="train",
                annotations=[
                    Bbox(0, 0, 1, 1, id=1, label=0),
                    Bbox(0, 0, 100, 100, id=2, label=0),
                ],
            ),
            DatasetItem(
                id="far_from_mean_1",
                subset="train",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=1),
                    Bbox(0, 0, 1e4, 1e4, id=2, label=1),
                ],
            ),
            DatasetItem(
                id="normal_test_0",
                subset="test",
                annotations=[
                    Bbox(0, 0, 100, 100, id=1, label=0),
                    Bbox(0, 0, 100, 100, id=2, label=1),
                ],
            ),
        ],
        categories=["a", "b"],
    )


@pytest.fixture
def fxt_refined_far_from_mean_dataset(fxt_original_far_from_mean_dataset):
    refined = deepcopy(fxt_original_far_from_mean_dataset)
    refined.put(
        DatasetItem(
            id="far_from_mean_0",
            subset="train",
            media=Image.from_numpy(data=np.ones((3, 2, 3))),
            annotations=[
                Bbox(0, 0, 100, 100, id=2, label=0),
            ],
        )
    )
    refined.put(
        DatasetItem(
            id="far_from_mean_1",
            subset="train",
            media=Image.from_numpy(data=np.ones((3, 2, 3))),
            annotations=[
                Bbox(0, 0, 100, 100, id=1, label=1),
            ],
        )
    )
    return refined


@pytest.fixture
def fxt_original_far_from_mean_attr_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="normal_train_0",
                subset="train",
                annotations=[
                    Bbox(
                        0, 0, 100, 100, label=0, attributes={"truncated": False, "occluded": False}
                    ),
                    Bbox(
                        0, 0, 100, 100, label=0, attributes={"truncated": False, "occluded": True}
                    ),
                    Bbox(
                        0, 0, 100, 100, label=1, attributes={"truncated": False, "occluded": False}
                    ),
                    Bbox(
                        0, 0, 100, 100, label=1, attributes={"truncated": False, "occluded": True}
                    ),
                ]
                * 100,
            ),
            DatasetItem(
                id="far_from_mean_0",
                subset="train",
                annotations=[
                    Bbox(
                        0,
                        0,
                        100,
                        100,
                        id=1,
                        label=0,
                        attributes={"truncated": False, "occluded": False},
                    ),
                    Bbox(
                        0,
                        0,
                        1,
                        1,
                        id=2,
                        label=0,
                        attributes={"truncated": False, "occluded": False},
                    ),
                    Bbox(
                        0, 0, 1, 1, id=3, label=1, attributes={"truncated": False, "occluded": True}
                    ),
                ],
            ),
            DatasetItem(
                id="far_from_mean_1",
                subset="train",
                annotations=[
                    Bbox(
                        0, 0, 1, 1, id=1, label=0, attributes={"truncated": False, "occluded": True}
                    ),
                    Bbox(
                        0,
                        0,
                        1,
                        1,
                        id=2,
                        label=1,
                        attributes={"truncated": False, "occluded": False},
                    ),
                    Bbox(
                        0,
                        0,
                        100,
                        100,
                        id=3,
                        label=1,
                        attributes={"truncated": False, "occluded": False},
                    ),
                ],
            ),
            DatasetItem(
                id="normal_test_0",
                subset="test",
                annotations=[
                    Bbox(
                        0,
                        0,
                        100,
                        100,
                        id=1,
                        label=0,
                        attributes={"truncated": False, "occluded": False},
                    ),
                    Bbox(
                        0,
                        0,
                        100,
                        100,
                        id=2,
                        label=1,
                        attributes={"truncated": False, "occluded": False},
                    ),
                ],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                [
                    ("a", "", ["truncated", "occluded"]),
                    ("b", "", ["truncated", "occluded"]),
                ]
            ),
        },
    )


@pytest.fixture
def fxt_refined_far_from_mean_attr_dataset(fxt_original_far_from_mean_attr_dataset):
    refined = deepcopy(fxt_original_far_from_mean_attr_dataset)
    refined.put(
        DatasetItem(
            id="far_from_mean_0",
            subset="train",
            media=Image.from_numpy(data=np.ones((3, 2, 3))),
            annotations=[
                Bbox(
                    0,
                    0,
                    100,
                    100,
                    id=1,
                    label=0,
                    attributes={"truncated": False, "occluded": False},
                ),
            ],
        )
    )
    refined.put(
        DatasetItem(
            id="far_from_mean_1",
            subset="train",
            media=Image.from_numpy(data=np.ones((3, 2, 3))),
            annotations=[
                Bbox(
                    0,
                    0,
                    100,
                    100,
                    id=3,
                    label=1,
                    attributes={"truncated": False, "occluded": False},
                ),
            ],
        )
    )
    return refined


@pytest.fixture
def fxt_original_cls_dataset():
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
def fxt_refined_cls_dataset(fxt_original_cls_dataset):
    refined = deepcopy(fxt_original_cls_dataset)
    refined.remove("missing_label_0", "train")
    refined.remove("multi_label_0", "train")
    refined.categories()[AnnotationType.label] = LabelCategories.from_iterable(
        ["label_" + str(label) for label in range(4)] + ["100"]
    )
    return refined


@pytest.fixture
def fxt_original_det_dataset():
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
def fxt_refined_det_dataset(fxt_original_det_dataset):
    refined = deepcopy(fxt_original_det_dataset)
    refined.put(
        DatasetItem(
            id="negative_len_0",
            subset="train",
            media=Image.from_numpy(data=np.ones((3, 2, 3))),
            annotations=[
                Bbox(0, 0, 100, 100, id=2, label=1),
            ],
        )
    )
    refined.put(
        DatasetItem(
            id="negative_len_1",
            subset="train",
            media=Image.from_numpy(data=np.ones((3, 2, 3))),
            annotations=[
                Bbox(0, 0, 100, 100, id=1, label=0),
            ],
        )
    )
    refined.put(
        DatasetItem(
            id="invalid_0",
            subset="train",
            media=Image.from_numpy(data=np.ones((3, 2, 3))),
            annotations=[
                Bbox(0, 0, 100, 100, id=1, label=0),
            ],
        )
    )
    refined.remove("missing_label_0", "train")
    refined.categories()[AnnotationType.label] = LabelCategories.from_iterable(
        ["label_" + str(label) for label in range(4)] + ["100"]
    )
    return refined


@pytest.fixture
def fxt_original_tabular_dataset():
    path = osp.join(get_test_asset_path("tabular_dataset"), "women_clothing_imbalanced.csv")
    tabular_dataset = Dataset.import_from(
        path,
        "tabular",
        target={
            "input": ["Review Text"],
            "output": [
                "Age",  # Caption
                "Title",  # Caption
                "Rating",  # Label
                "Positive Feedback Count",  # Caption
                "Division Name",  # Label
            ],
        },
    )
    tabular_dataset = transforms.AstypeAnnotations(tabular_dataset)

    # table = Table.from_csv(path)
    # table = Table.from_list([
    #     {"Rating": None, "Age": None, 'Review Text': None},
    #     {"Rating": 1, "Age": 10, 'Review Text': "Beautiful, stunning, cozy top!"},
    #     {"Rating": 1, "Age": 20, 'Review Text': "None"},
    #     {"Rating": 2, "Age": 30, 'Review Text': "Not impressed..."},
    #     {"Rating": 2, "Age": 40, 'Review Text': None},
    #     {"Rating": 3, "Age": 50, 'Review Text': None},
    #     {"Rating": 3, "Age": 60, 'Review Text': None},
    #     {"Rating": 4, "Age": 70, 'Review Text': None},
    #     {"Rating": 4, "Age": 80, 'Review Text': None},
    #     {"Rating": 5, "Age": 90, 'Review Text': None},
    #     {"Rating": 5, "Age": 100, 'Review Text': None},
    #     ])
    # dataset = Dataset.from_iterable(
    #     [
    #         DatasetItem(
    #             id="0",
    #             subset="train",
    #             media=TableRow(table=table, index=0),
    #             annotations=[],
    #         ),
    #         DatasetItem(
    #             id="1",
    #             subset="train",
    #             media=TableRow(table=table, index=1),
    #             annotations=[Label(label=0),],
    #         ),
    #         DatasetItem(
    #             id="2",
    #             subset="train",
    #             media=TableRow(table=table, index=2),
    #             annotations=[Label(label=0), Caption(label=1)],
    #         ),
    #     ],
    #     categories={
    #         AnnotationType.label: LabelCategories.from_iterable(
    #             [("Rating:1", "class"), ("Rating:2", "class"), ("Rating:3", "class"), ("Rating:4", "class"), ("Rating:5", "class")]
    #         )
    #     },
    #     media_type=TableRow,
    # )
    return tabular_dataset


@pytest.fixture
def fxt_refined_tabular_dataset():
    path = osp.join(get_test_asset_path("tabular_dataset"), "women_clothing_refined.csv")
    tabular_dataset = Dataset.import_from(
        path,
        "tabular",
        target={
            "input": ["Review Text"],
            "output": [
                "Age",
                "Title",
                "Rating",
                "Positive Feedback Count",
                "Division Name",
            ],
        },
    )
    tabular_dataset = transforms.AstypeAnnotations(tabular_dataset)
    return tabular_dataset


class ValidationCorrectionTest:
    @pytest.mark.parametrize(
        "fxt_original_dataset, fxt_validator, fxt_refined_dataset",
        [
            (
                "fxt_original_missing_cat_dataset",
                ClassificationValidator,
                "fxt_refined_missing_cat_dataset",
            ),
            (
                "fxt_original_missing_label_dataset",
                ClassificationValidator,
                "fxt_refined_missing_label_dataset",
            ),
            (
                "fxt_original_missing_attr_dataset",
                ClassificationValidator,
                "fxt_refined_missing_attr_dataset",
            ),
            (
                "fxt_original_multi_label_dataset",
                ClassificationValidator,
                "fxt_refined_multi_label_dataset",
            ),
            (
                "fxt_original_undefined_label_dataset",
                ClassificationValidator,
                "fxt_refined_undefined_label_dataset",
            ),
            (
                "fxt_original_undefined_attr_dataset",
                ClassificationValidator,
                "fxt_refined_undefined_attr_dataset",
            ),
            (
                "fxt_original_invalid_val_dataset",
                DetectionValidator,
                "fxt_refined_invalid_val_dataset",
            ),
            ("fxt_original_neg_len_dataset", DetectionValidator, "fxt_refined_neg_len_dataset"),
            (
                "fxt_original_far_from_mean_dataset",
                DetectionValidator,
                "fxt_refined_far_from_mean_dataset",
            ),
            (
                "fxt_original_far_from_mean_attr_dataset",
                DetectionValidator,
                "fxt_refined_far_from_mean_attr_dataset",
            ),
            ("fxt_original_cls_dataset", ClassificationValidator, "fxt_refined_cls_dataset"),
            ("fxt_original_det_dataset", DetectionValidator, "fxt_refined_det_dataset"),
        ],
        indirect=["fxt_original_dataset", "fxt_refined_dataset"],
        ids=[
            "missing_cat",
            "mising_label",
            "missing_attr",
            "multi_label",
            "undefined_label",
            "undefined_attr",
            "invalid_val",
            "neg_len",
            "far_from_mean",
            "far_from_mean_attr",
            "cls",
            "det",
        ],
    )
    def test_can_correct_data(
        self,
        fxt_original_dataset: Dataset,
        fxt_validator: Validator,
        fxt_refined_dataset: Dataset,
        request: pytest.FixtureRequest,
    ):
        validator = fxt_validator()
        reports = validator.validate(fxt_original_dataset)

        refined = transforms.Correct(fxt_original_dataset, reports=reports)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, refined, fxt_refined_dataset)

    @pytest.mark.parametrize(
        "fxt_original_dataset, fxt_validator, fxt_refined_dataset",
        [
            ("fxt_original_tabular_dataset", TabularValidator, "fxt_refined_tabular_dataset"),
        ],
        indirect=["fxt_original_dataset", "fxt_refined_dataset"],
        ids=[
            "tabular",
        ],
    )
    def test_can_correct_tabular_data(
        self,
        fxt_original_dataset: Dataset,
        fxt_validator: Validator,
        fxt_refined_dataset: Dataset,
        request: pytest.FixtureRequest,
    ):
        from datumaro.util import filter_dict, find

        validator = fxt_validator()
        reports = validator.validate(fxt_original_dataset)

        refined = transforms.Correct(fxt_original_dataset, reports=reports)

        helper_tc = request.getfixturevalue("helper_tc")
        # compare_datasets(helper_tc, refined, fxt_refined_dataset)
        for item_a in fxt_refined_dataset:
            item_b = find(refined, lambda x: x.id == item_a.id and x.subset == item_a.subset)
            self.assertFalse(item_b is None, item_a.id)

            self.assertEqual(len(item_a.annotations), len(item_b.annotations), item_a.id)
            for ann_a in item_a.annotations:
                # We might find few corresponding items, so check them all
                ann_b_matches = [x for x in item_b.annotations if x.type == ann_a.type]
                self.assertFalse(len(ann_b_matches) == 0, "ann id: %s" % ann_a.id)

                ann_b = find(
                    ann_b_matches,
                    lambda x: _compare_annotations(
                        x,
                        ann_a,
                    ),
                )
                if ann_b is None:
                    self.fail("ann %s, candidates %s" % (ann_a, ann_b_matches))
                item_b.annotations.remove(ann_b)  # avoid repeats
