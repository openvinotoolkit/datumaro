# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import Counter
from typing import List, Set
from unittest import TestCase

import numpy as np
import pytest

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Ellipse,
    Label,
    LabelCategories,
    Mask,
    Polygon,
)
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.errors import (
    AttributeDefinedButNotFound,
    DatasetValidationError,
    FarFromAttrMean,
    FarFromLabelMean,
    FewSamplesInAttribute,
    FewSamplesInLabel,
    ImbalancedAttribute,
    ImbalancedDistInAttribute,
    ImbalancedDistInLabel,
    ImbalancedLabels,
    InvalidValue,
    LabelDefinedButNotFound,
    MissingAnnotation,
    MissingAttribute,
    MissingLabelCategories,
    MultiLabelAnnotations,
    NegativeLength,
    OnlyOneAttributeValue,
    OnlyOneLabel,
    UndefinedAttribute,
    UndefinedLabel,
)
from datumaro.components.media import Image
from datumaro.components.validator import TaskType
from datumaro.plugins.configurable_validator import ConfigurableValidator

from ..requirements import Requirements, mark_requirement


@pytest.fixture
def fxt_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id=1,
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
                annotations=[
                    Label(
                        1,
                        id=0,
                        attributes={
                            "a": 1,
                            "b": 7,
                        },
                    ),
                    Bbox(
                        1,
                        2,
                        3,
                        4,
                        id=1,
                        label=0,
                        attributes={
                            "a": 1,
                            "b": 2,
                        },
                    ),
                    Mask(
                        id=2,
                        label=0,
                        attributes={"a": 1, "b": 2},
                        image=np.array(
                            [
                                [0, 0, 0, 0, 0],
                                [0, 0, 1, 1, 1],
                                [0, 0, 1, 1, 1],
                                [0, 0, 1, 1, 1],
                                [0, 0, 1, 1, 1],
                            ]
                        ),
                    ),
                ],
            ),
            DatasetItem(
                id=2,
                media=Image.from_numpy(data=np.ones((2, 4, 3))),
                annotations=[
                    Label(
                        2,
                        id=0,
                        attributes={
                            "a": 2,
                            "b": 2,
                        },
                    ),
                    Bbox(
                        2,
                        3,
                        1,
                        4,
                        id=1,
                        label=0,
                        attributes={
                            "a": 1,
                            "b": 1,
                        },
                    ),
                    Mask(
                        id=2,
                        label=0,
                        attributes={"a": 1, "b": 1},
                        image=np.array([[1, 1, 1, 1], [0, 0, 0, 0]]),
                    ),
                ],
            ),
            DatasetItem(id=3),
            DatasetItem(
                id=4,
                media=Image.from_numpy(data=np.ones((2, 4, 3))),
                annotations=[
                    Label(
                        0,
                        id=0,
                        attributes={
                            "b": 4,
                        },
                    ),
                    Label(
                        1,
                        id=1,
                        attributes={
                            "a": 11,
                            "b": 7,
                        },
                    ),
                    Bbox(
                        1,
                        3,
                        2,
                        4,
                        id=2,
                        label=0,
                        attributes={
                            "a": 2,
                            "b": 1,
                        },
                    ),
                    Bbox(
                        3,
                        1,
                        4,
                        2,
                        id=3,
                        label=0,
                        attributes={
                            "a": 2,
                            "b": 2,
                        },
                    ),
                    Polygon(
                        [1, 3, 1, 5, 5, 5, 5, 3],
                        label=0,
                        id=4,
                        attributes={
                            "a": 2,
                            "b": 2,
                        },
                    ),
                    Polygon(
                        [3, 1, 3, 5, 5, 5, 5, 1],
                        label=1,
                        id=5,
                        attributes={
                            "a": 2,
                            "b": 1,
                        },
                    ),
                ],
            ),
            DatasetItem(
                id=5,
                media=Image.from_numpy(data=np.ones((2, 4, 3))),
                annotations=[
                    Label(
                        0,
                        id=0,
                        attributes={
                            "a": 20,
                            "b": 10,
                        },
                    ),
                    Bbox(
                        1,
                        2,
                        3,
                        4,
                        id=1,
                        label=1,
                        attributes={
                            "a": 1,
                            "b": 1,
                        },
                    ),
                    Polygon(
                        [1, 2, 1, 5, 5, 5, 5, 2],
                        label=1,
                        id=2,
                        attributes={
                            "a": 1,
                            "b": 1,
                        },
                    ),
                ],
            ),
            DatasetItem(
                id=6,
                media=Image.from_numpy(data=np.ones((2, 4, 3))),
                annotations=[
                    Label(
                        1,
                        id=0,
                        attributes={
                            "a": 11,
                            "b": 2,
                            "c": 3,
                        },
                    ),
                    Bbox(
                        2,
                        3,
                        4,
                        1,
                        id=1,
                        label=1,
                        attributes={
                            "a": 2,
                            "b": 2,
                        },
                    ),
                    Mask(
                        id=2,
                        label=1,
                        attributes={"a": 2, "b": 2},
                        image=np.array(
                            [
                                [1, 0, 0],
                                [1, 0, 0],
                                [1, 0, 0],
                                [1, 0, 0],
                            ]
                        ),
                    ),
                ],
            ),
            DatasetItem(
                id=7,
                media=Image.from_numpy(data=np.ones((2, 4, 3))),
                annotations=[
                    Label(
                        1,
                        id=0,
                        attributes={
                            "a": 1,
                            "b": 2,
                            "c": 5,
                        },
                    ),
                    Bbox(
                        1,
                        2,
                        3,
                        4,
                        id=1,
                        label=2,
                        attributes={
                            "a": 1,
                            "b": 2,
                        },
                    ),
                    Polygon(
                        [1, 2, 1, 5, 5, 5, 5, 2],
                        label=2,
                        id=2,
                        attributes={
                            "a": 1,
                            "b": 2,
                        },
                    ),
                ],
            ),
            DatasetItem(
                id=8,
                media=Image.from_numpy(data=np.ones((2, 4, 3))),
                annotations=[
                    Label(
                        2,
                        id=0,
                        attributes={
                            "a": 7,
                            "b": 9,
                            "c": 5,
                        },
                    ),
                    Bbox(
                        2,
                        1,
                        3,
                        4,
                        id=1,
                        label=2,
                        attributes={
                            "a": 2,
                            "b": 1,
                        },
                    ),
                    Mask(
                        id=2,
                        label=2,
                        attributes={"a": 2, "b": 1},
                        image=np.array(
                            [
                                [1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1],
                            ]
                        ),
                    ),
                    Ellipse(
                        2,
                        1,
                        3,
                        4,
                        id=1,
                        label=2,
                        attributes={},
                    ),
                ],
            ),
            DatasetItem(
                id=9,
                media=Image.from_numpy(data=np.ones((2, 4, 3))),
                annotations=[
                    Bbox(
                        0,
                        0,
                        100,
                        100,
                        id=1,
                        label=0,
                        attributes={
                            "a": 100,
                            "b": 100,
                        },
                    ),
                    Polygon(
                        [0, 0, 0, 100, 100, 100, 100, 0],
                        label=0,
                        id=2,
                        attributes={
                            "a": 100,
                            "b": 100,
                        },
                    ),
                ],
            ),
            DatasetItem(
                id=10,
                media=Image.from_numpy(data=np.ones((2, 4, 3))),
                annotations=[
                    Bbox(
                        0,
                        0,
                        0.5,
                        0.5,
                        id=1,
                        label=0,
                        attributes={
                            "a": 100,
                            "b": 100,
                        },
                    ),
                    Bbox(
                        0,
                        0,
                        float("inf"),
                        10,
                        id=1,
                        label=1,
                        attributes={
                            "a": 100,
                            "b": 100,
                        },
                    ),
                    Polygon(
                        [1, 1, 1, 4, 4, 4, 4, 1],
                        label=0,
                        id=2,
                        attributes={
                            "a": 1,
                            "b": 1,
                        },
                    ),
                ],
            ),
            DatasetItem(
                id=11,
                media=Image.from_numpy(data=np.ones((2, 4, 3))),
                annotations=[
                    Bbox(
                        0,
                        0,
                        0.5,
                        0,
                        id=1,
                        label=0,
                        attributes={
                            "a": 2,
                            "b": 2,
                        },
                    ),
                    Bbox(
                        0,
                        0,
                        10,
                        float("nan"),
                        id=1,
                        label=1,
                        attributes={
                            "a": 1,
                            "b": 1,
                        },
                    ),
                    Polygon(
                        [1, 1, 1, 2, 2, 2, 2, 1],
                        label=1,
                        id=2,
                        attributes={
                            "a": 1,
                            "b": 1,
                        },
                    ),
                ],
            ),
        ],
        categories=[
            [
                f"label_{i}",
                None,
                {
                    "a",
                    "b",
                },
            ]
            for i in range(2)
        ],
    )


class ConfigurableValidatorTest:
    @pytest.mark.parametrize(
        "fxt_tasks,fxt_warnings",
        [
            ([TaskType.classification], {MissingAnnotation, MultiLabelAnnotations, UndefinedLabel}),
            (
                [TaskType.classification],
                {LabelDefinedButNotFound, OnlyOneLabel, FewSamplesInLabel, ImbalancedLabels},
            ),
            ([TaskType.classification], {MissingAttribute, UndefinedAttribute}),
            (
                [TaskType.classification],
                {
                    AttributeDefinedButNotFound,
                    OnlyOneAttributeValue,
                    FewSamplesInAttribute,
                    ImbalancedAttribute,
                },
            ),
            (
                [TaskType.detection],
                {MissingAnnotation, UndefinedLabel, OnlyOneLabel, FewSamplesInLabel},
            ),
            (
                [TaskType.detection],
                {FarFromLabelMean, ImbalancedDistInLabel},
            ),
            (
                [TaskType.detection],
                {InvalidValue, NegativeLength},
            ),
            (
                [TaskType.segmentation],
                {
                    AttributeDefinedButNotFound,
                    OnlyOneAttributeValue,
                    FewSamplesInAttribute,
                    ImbalancedAttribute,
                },
            ),
            (
                [TaskType.segmentation],
                {FarFromLabelMean, ImbalancedDistInLabel},
            ),
        ],
        ids=[
            "cls-label-anomaly",
            "cls-label",
            "cls-attr-anomaly",
            "cls-attr",
            "det-label",
            "det-pts",
            "det-anomaly",
            "seg-attr",
            "seg-pts",
        ],
    )
    def test_can_validate_configuration(
        self,
        fxt_dataset: Dataset,
        fxt_tasks: List[TaskType],
        fxt_warnings: Set[DatasetValidationError],
    ):
        validator = ConfigurableValidator(tasks=fxt_tasks, warnings=fxt_warnings, few_samples_thr=3)
        all_stats = validator.compute_statistics(fxt_dataset)
        all_reports = validator.generate_reports(all_stats)

        for task in fxt_tasks:
            stats = all_stats[task]

            reports = all_reports[task]
            reports = list(map(lambda r: r.to_dict(), reports))

            actual = set([r["anomaly_type"] for r in reports])
            expected = set([s.__name__ for s in fxt_warnings])

            assert actual.intersection(expected)
