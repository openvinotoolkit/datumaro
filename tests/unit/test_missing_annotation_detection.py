# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock

import numpy as np
import pytest

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset import Dataset, eager_mode
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image

from tests.requirements import Requirements, mark_requirement


class MissingAnnotationDetectionTest:
    @pytest.fixture
    def fxt_dataset(self) -> Dataset:
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id="item",
                    media=Image.from_numpy(np.zeros([2, 2, 3])),
                    annotations=[
                        Bbox(0, 0, 1, 1, label=0),
                        Bbox(1, 0, 1, 1, label=1),
                        Bbox(0, 1, 1, 1, label=2),
                    ],
                )
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [f"label_{label_id}" for label_id in range(3)]
                ),
            },
        )

    @pytest.fixture
    def fxt_launcher(self) -> MagicMock:
        gt_overlapped = Bbox(0, 0, 1, 1, label=1, attributes={"score": 1.0})
        missing_label = Bbox(1, 1, 1, 1, label=1, attributes={"score": 0.5})

        launcher_mock = MagicMock()
        launcher_mock.categories.return_value = None
        launcher_mock.launch.return_value = [[gt_overlapped, missing_label]]
        return launcher_mock

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("label_agnostic_matching, n_anns_expected", [(True, 1), (False, 2)])
    def test_label_matching_flags(
        self,
        fxt_dataset: Dataset,
        fxt_launcher: MagicMock,
        label_agnostic_matching: bool,
        n_anns_expected: int,
    ):
        with eager_mode():
            dataset = fxt_dataset.transform(
                "missing_annotation_detection",
                launcher=fxt_launcher,
                label_agnostic_matching=label_agnostic_matching,
            )

        item = dataset.get(id="item")

        assert len(item.annotations) == n_anns_expected

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("score_threshold, n_anns_expected", [(0.6, 0), (0.4, 1)])
    def test_score_threshold(
        self,
        fxt_dataset: Dataset,
        fxt_launcher: MagicMock,
        score_threshold: float,
        n_anns_expected: int,
    ):
        with eager_mode():
            dataset = fxt_dataset.transform(
                "missing_annotation_detection",
                launcher=fxt_launcher,
                score_threshold=score_threshold,
                label_agnostic_matching=True,
            )

        item = dataset.get(id="item")

        assert len(item.annotations) == n_anns_expected
