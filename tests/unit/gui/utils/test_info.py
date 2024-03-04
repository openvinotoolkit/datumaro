# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os

import pytest

from datumaro.components.annotation import LabelCategories
from datumaro.components.project import Dataset

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path

cwd = os.getcwd()

import sys

sys.path.append(os.path.join(cwd, "gui"))

from gui.datumaro_gui.utils.dataset.info import get_category_info, get_subset_info, return_matches


class InfoTest:
    @pytest.fixture
    def example_dataset(self):
        dataset_dir = get_test_asset_path("datumaro_dataset")
        dataset = Dataset.import_from(path=dataset_dir, format="datumaro")
        return dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_subset_info(self, example_dataset):
        result = sorted(get_subset_info(example_dataset), key=lambda x: x["id"])

        expected_result = [
            {"id": "test", "label": "test", "value": 1},
            {"id": "train", "label": "train", "value": 2},
            {"id": "validation", "label": "validation", "value": 1},
        ]

        assert all(x == y for x, y in zip(result, expected_result))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_category_info(self, example_dataset):
        example_categories = LabelCategories(
            items=[
                LabelCategories.Category(name="car", parent="", attributes=set()),
                LabelCategories.Category(name="bicycle", parent="", attributes=set()),
                LabelCategories.Category(name="tom", parent="", attributes=set()),
                LabelCategories.Category(name="mary", parent="", attributes=set()),
            ]
        )
        result = sorted(
            get_category_info(example_dataset, example_categories), key=lambda x: x["subset"]
        )

        expected_result = [
            {"car": 0, "bicycle": 1, "tom": 0, "mary": 1, "subset": "test"},
            {"car": 2, "bicycle": 1, "tom": 1, "mary": 1, "subset": "train"},
            {"car": 0, "bicycle": 0, "tom": 0, "mary": 0, "subset": "validation"},
        ]

        assert all(x == y for x, y in zip(result, expected_result))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_return_matches(self):
        # Test Case 1: Basic test with common elements
        first_labels = ["a", "b", "c"]
        second_labels = ["b", "c", "d"]
        first_name = "first"
        second_name = "second"
        expected_result = (["b", "c"], {"first": ["a"], "second": ["d"]})
        assert (
            return_matches(first_labels, second_labels, first_name, second_name) == expected_result
        )

        # Test Case 2: Lists with no common elements
        first_labels = ["a", "b", "c"]
        second_labels = ["d", "e", "f"]
        first_name = "first"
        second_name = "second"
        expected_result = ([], {"first": ["a", "b", "c"], "second": ["d", "e", "f"]})
        assert (
            return_matches(first_labels, second_labels, first_name, second_name) == expected_result
        )

        # Test Case 3: One empty list
        first_labels = ["a", "b", "c"]
        second_labels = []
        first_name = "first"
        second_name = "second"
        expected_result = ([], {"first": ["a", "b", "c"], "second": []})
        assert (
            return_matches(first_labels, second_labels, first_name, second_name) == expected_result
        )
